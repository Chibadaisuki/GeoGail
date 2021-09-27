from gail import gail
from env import timegeo_env
import yaml
import numpy as np


if __name__ == '__main__':
    env = timegeo_env()
    fake_file = np.loadtxt('./raw_data/geolife/fake.data')
    file = np.loadtxt('./raw_data/geolife/real.data')

    test = gail(
        env=env,
        file=file,
        fake_file=fake_file
    )
    test.run()
