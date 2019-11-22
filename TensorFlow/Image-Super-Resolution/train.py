from ISR.models import RRDN, RDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from ISR.train import Trainer
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    lr_train_patch_size = 40
    layers_to_extract = [5, 9]
    scale = 2
    hr_train_patch_size = lr_train_patch_size * scale

    # RRDN model
    rrdn = RRDN(arch_params={'C': 4, 'D': 3, 'G': 64, 'G0': 64, 'T': 10, 'x': scale}, patch_size=lr_train_patch_size)
    # RDN model
    rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2}, patch_size=lr_train_patch_size)

    f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
    discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

    loss_weights = {
        'generator': 0.0,
        'feature_extractor': 0.0833,
        'discriminator': 0.01
    }
    losses = {
        'generator': 'mae',
        'feature_extractor': 'mse',
        'discriminator': 'binary_crossentropy'
    }

    log_dirs = {'logs': './logs', 'weights': './weights'}

    learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

    flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

    trainer = Trainer(
        generator=rdn,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir='data/low_res/training/',
        hr_train_dir='data/high_res/training/',
        lr_valid_dir='data/low_res/validation/',
        hr_valid_dir='data/high_res/validation/',
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname='image_dataset',
        log_dirs=log_dirs,
        weights_generator=None,
        weights_discriminator=None,
        n_validation=2,
    )

    trainer.train(
        epochs=2,
        steps_per_epoch=2,
        batch_size=2,
        monitored_metrics={'val_loss': 'min'}
    )
