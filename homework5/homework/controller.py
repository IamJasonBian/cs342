import pystk


import pystk
import numpy as np

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    # Set the target velocity
    target_velocity = 20

    # Control the kart acceleration
    if current_vel < target_velocity:
        action.acceleration = 1  # accelerate if current velocity is less than the target
    else:
        action.acceleration = 0  # stop accelerating if we've reached the target velocity

    # Control the kart braking
    if current_vel > target_velocity:
        action.brake = True  # apply brakes if current velocity is more than the target
    else:
        action.brake = False  # release brakes if we're below the target velocity

    # Control the kart steering
    action.steer = np.clip(aim_point[0], -1, 1)

    # Control the kart drifting
    if np.abs(action.steer) > 0.5:
        action.drift = True  # start drifting for wide turns
    else:
        action.drift = False  # stop drifting for narrower turns

    return action



if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
