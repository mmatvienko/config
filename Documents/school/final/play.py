import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse
import pystk
from time import time, sleep
import numpy as np
from . import gui


def action_dict(action):
    return {k: getattr(action, k) for k in ['acceleration', 'brake', 'steer', 'fire', 'drift']}

def get_vector_from_this_to_that(me, obj, normalize=True):
    """
    Expects numpy arrays as input
    """
    vector = obj - me

    if normalize:
        return vector / np.linalg.norm(vector)

    return vector


def to_numpy(location):
    """
    Don't care about location[1], which is the height
    """
    return np.float32([location[0], location[2]])


if __name__ == "__main__":
    soccer_tracks = {"soccer_field", "icy_soccer_field"}

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--track')
    parser.add_argument('-k', '--kart', default='')
    parser.add_argument('--team', type=int, default=0, choices=[0, 1])
    parser.add_argument('-s', '--step_size', type=float)
    parser.add_argument('-n', '--num_player', type=int, default=1)
    parser.add_argument('-v', '--visualization', type=str, choices=list(gui.VT.__members__), nargs='+',
                        default=['IMAGE'])
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_dir', type=Path, required=False)
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    config = pystk.GraphicsConfig.hd()
    config.screen_width = 128
    config.screen_height = 96
    pystk.init(config)

    config = pystk.RaceConfig()
    config.num_kart = 2
    if args.kart is not None:
        config.players[0].kart = args.kart


    config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    config.players[0].team = args.team

    for i in range(1, args.num_player):
        config.players.append(
                pystk.PlayerConfig(args.kart, pystk.PlayerConfig.Controller.AI_CONTROL, (args.team + 1) % 2))

    if args.track is not None:
        config.track = args.track
        if args.track in soccer_tracks:
            config.mode = config.RaceMode.SOCCER
    if args.step_size is not None:
        config.step_size = args.step_size

    race = pystk.Race(config)
    race.start()

    uis = [gui.UI([gui.VT[x] for x in args.visualization]) for i in range(args.num_player)]

    state = pystk.WorldState()
    t0 = time()
    n = 0
    ax = plt.gcf().add_subplot(3, 3, 9)

    while all(ui.visible for ui in uis):
        if not all(ui.pause for ui in uis):
            race.step(uis[0].current_action)
            state.update()

            if args.verbose and config.mode == config.RaceMode.SOCCER:
                print('Score ', state.soccer.score)
                print('      ', state.soccer.ball)
                print('      ', state.soccer.goal_line)

        for ui, d in zip(uis, race.render_data):
            ui.show(d)

        if args.save_dir:
            pos_ball = to_numpy(state.soccer.ball.location)
            pos_me = to_numpy(state.karts[1].location)
            front_me = to_numpy(state.karts[1].front)

            ori_me = get_vector_from_this_to_that(pos_me, front_me)
            ori_to_puck = get_vector_from_this_to_that(pos_me, pos_ball)

            # rotate ori_to_puck counter clock wise and dot to get condition for side
            dot_cond = -1* ori_me[0] * ori_to_puck[1] + ori_me[1] * ori_to_puck[0]
            angle = np.arccos(np.dot(ori_me, ori_to_puck)/(np.linalg.norm(ori_me)*np.linalg.norm(ori_to_puck)))
            print(dot_cond)
            angle = angle if dot_cond > 0 else angle*-1
            
            image = np.array(race.render_data[0].image)
            action = action_dict(uis[0].current_action)
            print(f'angle: {angle*(180/np.pi)}')
            ax.clear()
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            
            ax.plot(pos_me[0], pos_me[1], 'r.')                 # Current player is a red dot.
            ax.plot(pos_ball[0], pos_ball[1], 'co')             # The puck is a cyan circle.
            
            # Plot lines of where I am facing, and where the enemy is in relationship to me.
            ax.plot([pos_me[0], pos_me[0] + 10 * ori_me[0]], [pos_me[1], pos_me[1] + 10 * ori_me[1]], 'r-')
            ax.plot([pos_me[0], pos_me[0] + 10 * ori_to_puck[0]], [pos_me[1], pos_me[1] + 10 * ori_to_puck[1]], 'b-')
            
            # Live debugging of scalars. Angle in degrees to the target item.
            ax.set_title('%.2f' % (angle*(180/np.pi)))
            

            
            Image.fromarray(image).save(args.save_dir / ('image_%06d.png' % n))
            (args.save_dir / ('action_%06d.txt' % n)).write_text(str(action) + "\n" + str(pos_ball))

            # Make sure we play in real time
        n += 1
        delta_d = n * config.step_size - (time() - t0)
        if delta_d > 0:
            ui.sleep(delta_d)
        
    race.stop()
    del race
    pystk.clean()
