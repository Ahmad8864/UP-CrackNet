import torch
from torchvision import transforms
from torchvision.utils import make_grid 
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator, Discriminator
import argparse
import os
from math import exp
import torch.nn.functional as F
from msgms_loss import MSGMS_Loss
import datetime
from torch.utils.tensorboard import SummaryWriter 

from PS_loss import StyleLoss, PerceptualLoss

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='/kaggle/input/up-cracknet/road/road', help='input dataset')
parser.add_argument('--batch_size', type=int, default=8, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=300, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0008, help='learning rate for generator, default=0.0008')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)

logdir = './path/log_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(logdir)

# SSIM:
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


DEBUG_TB = True
def _to01_from_norm(x: torch.Tensor) -> torch.Tensor:
    # inputs/targets normalized with mean=0.5, std=0.5
    return x.mul(0.5).add(0.5).clamp_(0.0, 1.0)

def _to01_from_tanh(x: torch.Tensor) -> torch.Tensor:
    # generator outputs with tanh in [-1,1]
    return x.add(1.0).mul_(0.5).clamp_(0.0, 1.0)

def _prep_mask_for_vis(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Return mask as (B,3,H,W) float32 in [0,1], contiguous.
    Accepts (B,H,W), (B,1,H,W), or (B,3,H,W) in any numeric/bool dtype.
    """
    m = mask
    if m.dim() == 3:              # (B,H,W)
        m = m.unsqueeze(1)        # (B,1,H,W)
    if m.size(1) == 1:            # (B,1,H,W) -> (B,3,H,W)
        m = m.repeat(1, 3, 1, 1)
    elif m.size(1) != 3:
        # if someone passed weird channels, just take the first and repeat
        m = m[:, :1].repeat(1, 3, 1, 1)

    m = m.to(torch.float32)       # convert dtype
    # normalize to [0,1] if not already binary
    m_min = m.amin(dim=(1,2,3), keepdim=True)
    m_max = m.amax(dim=(1,2,3), keepdim=True)
    span  = (m_max - m_min).clamp_min(1e-6)
    m = (m - m_min) / span
    # ensure the spatial size matches (H,W) in case mask came in different size
    if m.shape[-2:] != (H, W):
        m = torch.nn.functional.interpolate(m, size=(H, W), mode="nearest")
    return m.contiguous()

def log_tb_samples(writer, x_in, y_tgt, y_pred, mask, epoch, tag_prefix="samples", max_samples=4):
    """
    x_in:   (B,3,H,W) normalized by mean=0.5,std=0.5
    y_tgt:  (B,3,H,W) same normalization
    y_pred: (B,3,H,W) tanh output
    mask:   (B,H,W) or (B,1,H,W) or (B,3,H,W), any dtype
    """
    B, C, H, W = x_in.shape
    n = min(max_samples, B)

    # Slice and move to CPU for TensorBoard
    x_vis    = _to01_from_norm(x_in[:n].detach().cpu())
    y_vis    = _to01_from_norm(y_tgt[:n].detach().cpu())
    pred_vis = _to01_from_tanh(y_pred[:n].detach().cpu())
    m_vis    = _prep_mask_for_vis(mask[:n].detach().cpu(), H=H, W=W)

    # Build grids (CHW floats in [0,1])
    grid_kwargs = dict(nrow=n, padding=2, pad_value=0.5)
    grid_input   = make_grid(x_vis,    **grid_kwargs).contiguous()
    grid_target  = make_grid(y_vis,    **grid_kwargs).contiguous()
    grid_output  = make_grid(pred_vis, **grid_kwargs).contiguous()
    grid_mask    = make_grid(m_vis,    **grid_kwargs).contiguous()

    if DEBUG_TB:
        print(
            "[TB] input",  tuple(grid_input.shape),  grid_input.dtype,
            "| target",    tuple(grid_target.shape), grid_target.dtype,
            "| output",    tuple(grid_output.shape), grid_output.dtype,
            "| mask",      tuple(grid_mask.shape),   grid_mask.dtype
        )

    # Force CHW dataformat so TB/PIL won't mis-infer HWC
    writer.add_image(f"{tag_prefix}/input",  grid_input,  global_step=epoch, dataformats="CHW")
    writer.add_image(f"{tag_prefix}/target", grid_target, global_step=epoch, dataformats="CHW")
    writer.add_image(f"{tag_prefix}/mask",   grid_mask,   global_step=epoch, dataformats="CHW")
    writer.add_image(f"{tag_prefix}/output", grid_output, global_step=epoch, dataformats="CHW")


data_dir = params.dataset

# Create a timestamped subfolder inside saved-logs
base_log_dir = './saved-logs/'
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = os.path.join(base_log_dir, timestamp)
if not os.path.exists(base_log_dir):
    os.mkdir(base_log_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Set up log file in model_dir
log_file_path = os.path.join(model_dir, "train_log.txt")
def log_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    with open(log_file_path, "a") as f:
        f.write(msg + "\n")

transform = transforms.Compose([transforms.Resize(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


train_data = DatasetFromFolder(data_dir, subfolder='train', resize_scale=params.resize_scale,  transform=transform, crop_size=params.crop_size, fliplr=params.fliplr)


train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True, pin_memory=True, num_workers=72, prefetch_factor=20, persistent_workers=True)

test_data = DatasetFromFolder(data_dir, subfolder='validation', resize_scale=params.resize_scale,  transform=transform, crop_size=params.crop_size)

test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)

# test_input, test_target, test_mask = test_data_loader.__iter__().__next__()


G = Generator(3, params.ngf, 3)
D = Discriminator(6, params.ndf, 1)
G.cuda()
D.cuda()

G.normal_weight_init(mean=0.0, std=0.02)
D.normal_weight_init(mean=0.0, std=0.02)

BCE_loss = torch.nn.BCEWithLogitsLoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
L2_loss = torch.nn.MSELoss().cuda()


perceptual_loss = PerceptualLoss().cuda()
style_loss = StyleLoss().cuda()
msgms_loss = MSGMS_Loss().cuda()

G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

def adjust_learning_rate1(optimizer, epoch):
    lr = params.lrG*(0.99**(epoch))
    print("G lr is {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(optimizer, epoch):
    lr = params.lrD*(0.99**(epoch))
    print("D lr is {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def g_adv_scale(epoch):
    return 5 if epoch < 10 else 4 * (2/3) ** (epoch - 10) + 1

D_avg_losses = []
G_avg_losses = []

step = 0

loss_L1 = False
loss_L1_Style = False
loss_L1_SSIM_GMS = False
loss_L1_SSIM_GMS_Style = True

best_val_loss = 10000000000

for epoch in range(params.num_epochs):
    D_losses = []
    G_losses = []
    adjust_learning_rate1(G_optimizer, epoch)
    adjust_learning_rate2(D_optimizer, epoch)

    for i, (input, target, mask) in enumerate(train_data_loader):
        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())
        m_ = Variable(mask.cuda())

        D_real_decision = D(x_, y_).view(-1)
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)

        with torch.no_grad():
            gen_image_d = G(x_)
        D_fake_decision = D(x_, gen_image_d).view(-1)
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).view(-1)
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        if loss_L1:
            # using pure L1 loss
            l1_loss = params.lamb * L1_loss(gen_image, y_)
        elif loss_L1_SSIM_GMS_Style:
            loss_MSGMS = msgms_loss(gen_image, y_)
            loss_SSIM = 1 - ssim(gen_image, y_)
            gen_style_loss = style_loss(gen_image, y_) * 10
            l_rec = gen_style_loss + loss_MSGMS + loss_SSIM + L1_loss(gen_image, y_)
            l1_loss = params.lamb * l_rec

        # g_adv_scale(i) * G_fake_loss + l1_loss
        G_loss = G_fake_loss + l1_loss
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        log_print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.item(), G_loss.item()))
        log_print("")
        step += 1

    writer.add_scalar('G_loss_mean', torch.mean(torch.FloatTensor(G_losses)), epoch)
    writer.add_scalar('D_loss_mean', torch.mean(torch.FloatTensor(D_losses)), epoch)
    # right after computing each component
    writer.add_scalar("loss/D_total", D_loss.item(), epoch)
    writer.add_scalar("loss/G_total", G_loss.item(), epoch)
    writer.add_scalar("loss/G_adv", G_fake_loss.item(), epoch)
    writer.add_scalar("loss/G_L1", L1_loss(gen_image, y_).item(), epoch)
    writer.add_scalar("loss/G_SSIM", (1 - ssim(gen_image, y_)).item(), epoch)
    writer.add_scalar("loss/G_MSGMS", msgms_loss(gen_image, y_).item(), epoch)
    writer.add_scalar("loss/G_stylex10", (style_loss(gen_image, y_) * 10).item(), epoch)

    # also track D(real) and D(fake) outputs (after sigmoid if using BCE)
    with torch.no_grad():
        writer.add_scalar("score/D_real_mean", D_real_decision.mean().item(), epoch)
        writer.add_scalar("score/D_fake_mean", D_fake_decision.mean().item(), epoch)

    if epoch % 1 == 0:   # log every epoch
        with torch.no_grad():
            gen_sample = G(x_)
            log_tb_samples(writer, x_in=x_, y_tgt=y_, y_pred=gen_image, mask=m_, epoch=epoch, tag_prefix="train", max_samples=4)

    if (epoch+1) % 10 == 0: 
        val_losses = 0.00
        # time_start = time.time()
        for i, (input, target, mask) in enumerate(test_data_loader):
            x_ = Variable(input.cuda())
            y_ = Variable(target.cuda())
            
            with torch.no_grad():
                gen_image = G(x_)
            if loss_L1:
                # using pure L1 loss
                l1_loss =  L1_loss(gen_image, y_)
            elif loss_L1_SSIM_GMS_Style:
                loss_MSGMS = msgms_loss(gen_image, y_)
                loss_SSIM = 1 - ssim(gen_image, y_)
                gen_style_loss = style_loss(gen_image, y_) * 10
                l_rec = gen_style_loss + loss_MSGMS + loss_SSIM + L1_loss(gen_image, y_)
                l1_loss = params.lamb * l_rec

            loss_all = l1_loss
            val_losses += loss_all

        if val_losses < best_val_loss:
            best_val_loss = min(best_val_loss, val_losses)
            log_print("best_val_loss is {}".format(best_val_loss))
            torch.save(G.state_dict(), os.path.join(model_dir, 'best_G_param.pkl'))
            torch.save(D.state_dict(), os.path.join(model_dir, 'best_D_param.pkl'))
            log_print("the best model is epoch_{}".format(epoch + 1))

    # utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)
    if (epoch+1) % 10 == 0:
        torch.save(G.state_dict(), os.path.join(model_dir, f'{epoch + 1}_generator_param.pkl'))
        torch.save(D.state_dict(), os.path.join(model_dir, f'{epoch + 1}_discriminator_param.pkl'))
    

