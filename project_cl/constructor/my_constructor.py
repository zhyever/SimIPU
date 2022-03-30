from mmcv.utils import build_from_cfg

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS

from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor

@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(DefaultOptimizerConstructor):

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        super(MyOptimizerConstructor, self).__init__(optimizer_cfg, paramwise_cfg)

    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        predictor_prefix = ('module.predictor', 'predictor')
        parameters = [{
            'name': 'base',
            'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
            'lr': self.optimizer_cfg['lr']
        },{
            'name': 'predictor',
            'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
            'lr': self.optimizer_cfg['lr']
        }]

        optimizer_cfg = self.optimizer_cfg.copy()

        # set param-wise lr and weight decay recursively
        params = parameters
        # self.add_params(params, model)
        optimizer_cfg['params'] = params
        
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
