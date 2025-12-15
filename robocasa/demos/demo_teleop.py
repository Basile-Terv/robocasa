import argparse
import json
import time
from collections import OrderedDict

import robocasa.macros as macros

import robosuite
from pynput.keyboard import Controller, Key, Listener
from robocasa.scripts.collect_demos import (
    collect_human_trajectory,
    gather_demonstrations_as_hdf5,
)
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from termcolor import colored


def choose_option(
    options, option_name, show_keys=False, default=None, default_message=None
):
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all tasks

    if default is None:
        default = options[0]

    if default_message is None:
        default_message = default

    # Select environment to run
    print("Here is a list of {}s:\n".format(option_name))

    for i, (k, v) in enumerate(options.items()):
        if show_keys:
            print("[{}] {}: {}".format(i, k, v))
        else:
            print("[{}] {}".format(i, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1,
                default_message,
            )
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except:
        if default is None:
            choice = options[0]
        else:
            choice = default
        print("Use {} by default.\n".format(choice))

    # Return the chosen environment name
    return choice


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="task (choose among 100+ tasks)")
    parser.add_argument("--layout", type=int, help="kitchen layout (choose number 0-9)")
    parser.add_argument("--style", type=int, help="kitchen style (choose number 0-11)")
    parser.add_argument(
        "--device", type=str, default="keyboard", choices=["keyboard", "spacemouse"]
    )
    parser.add_argument("--robot", type=str, help="robot")
    args = parser.parse_args()

    tasks = OrderedDict(
        [
            ("PnPCounterToCab", "pick and place from counter to cabinet"),
            ("PnPCounterToSink", "pick and place from counter to sink"),
            ("PnPMicrowaveToCounter", "pick and place from microwave to counter"),
            ("PnPStoveToCounter", "pick and place from stove to counter"),
            ("OpenSingleDoor", "open cabinet or microwave door"),
            ("CloseDrawer", "close drawer"),
            ("TurnOnMicrowave", "turn on microwave"),
            ("TurnOnSinkFaucet", "turn on sink faucet"),
            ("TurnOnStove", "turn on stove"),
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
            ("PnPCounterTop", "pick and place"),
        ]
    )

    if args.task is None:
        args.task = choose_option(
            tasks, "task", default="PnPCounterToCab", show_keys=True
        )
    robots = OrderedDict([(0, "TMR_ROBOT"), (1, "PandaOmron")])

    if args.robot is None:
        robot_choice = choose_option(
            robots, "robot", default=0, default_message="TMR_ROBOT"
        )
        args.robot = robots[robot_choice]

    # Create argument configuration
    print("THE ROBOT IS: ", args.robot)
    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
        # "layout_ids": args.layout,
        # "style_ids": args.style,
        "translucent_robot": False,
    }

    args.renderer = "mjviewer"

    print(colored(f"Initializing environment...", "yellow"))
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_robotview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=10,
        renderer=args.renderer,
        camera_heights=300,  # set camera height
        camera_widths=480,  # set camera width
        camera_names=[
            "robot0_robotview",
            "robot0_leftview",
            "robot0_rightview",
        ],  # use "agentview" camera
        camera_depths=True,
        use_distractors=True,
        mode=1,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    else:
        raise ValueError
    tmp_directory = "./robocasa_data/teleop_data/kitchen/driod_0922_batch5"
    env = DataCollectionWrapper(env, tmp_directory)
    recording_enabled = False

    def _on_key_press(key):
        """
        Handles key press events to toggle recording.
        """
        global recording_enabled
        try:
            if key.char == "v":
                recording_enabled = not recording_enabled
                if recording_enabled:
                    print("Recording started...")
                else:
                    print("Recording stopped. Flushing data...")
                    env._flush()
        except AttributeError:
            pass

    listener = Listener(on_press=_on_key_press)
    listener.start()
    # # collect demonstrations
    excluded_eps = []
    try:
        while True:
            ep_directory, discard_traj = collect_human_trajectory(
                env,
                device,
                "right",
                "TwoArm",
                mirror_actions=True,
                render=(args.renderer != "mjviewer"),
                max_fr=30,
            )
            if recording_enabled:
                print("Recording started and ENABLED")

                path = gather_demonstrations_as_hdf5(
                    ep_directory, tmp_directory, env_info
                )
                excluded_eps.append(ep_directory.split("/")[-1])
                convert_to_robomimic_format(path)

            print()
    except KeyboardInterrupt:
        print("\nInterrupted. Saving and exiting.")
        exit(0)
