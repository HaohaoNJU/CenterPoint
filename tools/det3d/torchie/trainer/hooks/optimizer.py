from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad
from .hook import Hook


class OptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        trainer.optimizer.zero_grad()
        # print(trainer.outputs["loss"])
        trainer.outputs["loss"].backward()
        if self.grad_clip is not None:
            self.clip_grads(trainer.model.parameters())
        trainer.optimizer.step()

class AmpOptimizerHook(Hook):
    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip
        self.scaler = GradScaler()
    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip
        )

    def after_train_iter(self, trainer):
        # print("mix prec is on the go   !   !   !")
        trainer.optimizer.zero_grad()
        # print(trainer.outputs["loss"])
        self.scaler.scale(trainer.outputs['loss']).backward()
        self.scaler.step(trainer.optimizer)
        self.scaler.update()

    # def amp_after_train_iter(self, outputs,scaler):
