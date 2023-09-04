from utils.options import Options
from utils.trainer import Trainer


def run(options):
    Trainer(
        dataset_params=options.dataset_params,
        general_params=options.general_params,
        trainer_params=options.trainer_params,
        model_params=options.model_params,
    ).run()


if __name__ == "__main__":
    run(Options().parse_args())
