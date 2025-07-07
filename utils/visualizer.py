import os
import time
import torch
from utils.utils import tensor2uint
import matplotlib.pyplot as plt
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags;
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0
        self.experimrnt_dir = opt.experimrnt_dir
        self.checkpoints_dir = opt.checkpoints_dir
        if self.use_wandb:
            self.wandb_run = wandb.init(project='self-denosie', name=opt.name,config=opt) if not wandb.run else wandb.run
            # self.wandb_run._label(repo='IRVDGAN')
        else:
            self.loss_list = {}
            self.loss_list['total'] = []
            for loss in opt.loss_type:
                self.loss_list[loss] = []
        # create a logging file to store training losses
        self.log_name = os.path.join(self.experimrnt_dir, 'log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch):
        """Display current results on visdom; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
        """
        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = tensor2uint(image)
                wandb_image = wandb.Image(image_numpy)
                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})

    def plot_current_losses(self, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if self.use_wandb:
            self.wandb_run.log(losses)
        else: 

            plt.figure()
            self.loss_list['total'].append(losses['total'].item()) 
            steps = range(1, len(self.loss_list['total']) + 1, 1)
            plt.plot(steps, self.loss_list['total'], label='total')            
            for loss in self.opt.loss_type:
                self.loss_list[loss].append(losses[loss].item())                      
                plt.plot(steps, self.loss_list[loss], label=loss)  
            plt.title("Loss Curves")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(self.experimrnt_dir, 'loss.png'))
            plt.close()            
    def plot_current_metrics(self, psnr, ssim):
        if self.use_wandb:
            self.wandb_run.log({
                "psnr": psnr,
                "ssim": ssim})

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d) ' % (epoch, iters)

        for k, v in losses.items():
            message += '%s: %.8f ' % (k, v)    
        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_logs(self, message, do_print=False):
        """打印logs并保存到txt"""
        if do_print:
            print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
