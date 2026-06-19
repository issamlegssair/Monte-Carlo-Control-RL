"""
=============================================================
DEMO LIVE — Monte Carlo Control
=============================================================
Lance la demo et sauvegarde automatiquement en GIF + MP4.

UTILISATION :
  pip install numpy matplotlib pillow
  python demo_live.py
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # pas de fenetre : on enregistre directement
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from collections import defaultdict

np.random.seed(42)

# =============================================================
# ENVIRONNEMENT
# =============================================================

class GridWorld:
    def __init__(self, size=5, gamma=0.9):
        self.size  = size
        self.gamma = gamma
        self.goal  = (size-1, size-1)
        self.walls = {(1,1),(1,2),(2,1)}
        self.state = None
        self.reset()

    def reset(self):
        while True:
            s = (np.random.randint(self.size),
                 np.random.randint(self.size))
            if s != self.goal and s not in self.walls:
                self.state = s
                return s

    def step(self, action):
        moves = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        r, c   = self.state
        dr, dc = moves[action]
        nr, nc = r+dr, c+dc
        if (0<=nr<self.size and 0<=nc<self.size
                and (nr,nc) not in self.walls):
            self.state = (nr,nc)
        done   = (self.state == self.goal)
        reward = 10.0 if done else -1.0
        return self.state, reward, done


# =============================================================
# AGENT
# =============================================================

class MonteCarloAgent:
    def __init__(self, n_actions=4, gamma=0.9, epsilon=0.4):
        self.n_actions = n_actions
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.N = defaultdict(lambda: np.zeros(n_actions))

    def choisir_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def generer_episode(self, env, max_steps=150):
        episode, state, done, steps = [], env.reset(), False, 0
        trajectory = [state]
        while not done and steps < max_steps:
            a = self.choisir_action(state)
            ns, r, done = env.step(a)
            episode.append((state, a, r))
            trajectory.append(ns)
            state = ns; steps += 1
        return episode, trajectory

    def mettre_a_jour(self, episode):
        G, visited = 0.0, set()
        for state, action, reward in reversed(episode):
            G  = self.gamma * G + reward
            sa = (state, action)
            if sa in visited: continue
            visited.add(sa)
            self.N[state][action] += 1
            n = self.N[state][action]
            self.Q[state][action] += (G - self.Q[state][action]) / n

    def valeur_etat(self, size):
        V = np.zeros((size, size))
        for r in range(size):
            for c in range(size):
                V[r,c] = np.max(self.Q[(r,c)])
        return V

    def obtenir_politique(self, env):
        pol = {}
        for r in range(env.size):
            for c in range(env.size):
                s = (r,c)
                if s != env.goal and s not in env.walls:
                    pol[s] = int(np.argmax(self.Q[s]))
        return pol


# =============================================================
# PRE-ENTRAINEMENT : generer tous les frames a l'avance
# =============================================================

def pre_entrainer(n_episodes=400, capture_every=5):
    """
    Entraine l'agent et capture un snapshot tous les
    capture_every episodes. Retourne la liste des snapshots.
    """
    env   = GridWorld(size=5, gamma=0.9)
    agent = MonteCarloAgent(epsilon=0.4)

    WINDOW       = 30
    hist_rewards = []
    hist_smooth  = []
    snapshots    = []

    print(f"Pre-entrainement : {n_episodes} episodes...")

    for ep in range(1, n_episodes+1):
        episode, trajectory = agent.generer_episode(env)
        agent.mettre_a_jour(episode)
        total_r = sum(r for _,_,r in episode)
        hist_rewards.append(total_r)
        if len(hist_rewards) >= WINDOW:
            hist_smooth.append(np.mean(hist_rewards[-WINDOW:]))

        agent.epsilon = max(0.02, agent.epsilon * 0.997)

        if ep % capture_every == 0 or ep == 1:
            snapshots.append({
                'ep'         : ep,
                'epsilon'    : agent.epsilon,
                'politique'  : agent.obtenir_politique(env),
                'V'          : agent.valeur_etat(env.size).copy(),
                'trajectory' : list(trajectory),
                'reward'     : total_r,
                'hist'       : list(hist_rewards),
                'smooth'     : list(hist_smooth),
                'n_explored' : len(agent.Q),
            })

        if ep % 50 == 0:
            moy = np.mean(hist_rewards[-50:])
            print(f"  Episode {ep:4d}/{n_episodes} | "
                  f"Moy recompense : {moy:6.2f} | "
                  f"epsilon : {agent.epsilon:.3f}")

    print(f"Snapshots captures : {len(snapshots)}")
    return snapshots, env


# =============================================================
# CONSTRUCTION DE L'ANIMATION
# =============================================================

def construire_animation(snapshots, env):
    SIZE    = env.size
    WINDOW  = 30
    arrows  = {0:'^', 1:'v', 2:'<', 3:'>'}
    N_TOTAL = snapshots[-1]['ep']

    # Palette sombre
    C = {
        'bg'     : '#0d1117',
        'panel'  : '#161b22',
        'border' : '#30363d',
        'normal' : '#21262d',
        'wall'   : '#373e47',
        'goal'   : '#196c2e',
        'path'   : '#7d2a0e',
        'robot'  : '#6e40c9',
        'arrow'  : '#e6edf3',
        'text'   : '#e6edf3',
        'muted'  : '#8b949e',
        'yellow' : '#f0e68c',
        'orange' : '#f78166',
        'green'  : '#3fb950',
        'purple' : '#bc8cff',
    }

    # ----- Mise en page -----
    fig = plt.figure(figsize=(14, 8), facecolor=C['bg'])
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.05, right=0.97,
        top=0.88, bottom=0.09
    )

    ax_grid = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[0, 2])
    ax_conv = fig.add_subplot(gs[1, :2])
    ax_info = fig.add_subplot(gs[1, 2])

    for ax in [ax_grid, ax_heat, ax_traj, ax_conv, ax_info]:
        ax.set_facecolor(C['panel'])
        for sp in ax.spines.values():
            sp.set_edgecolor(C['border'])

    fig.suptitle(
        'Monte Carlo Control  —  Apprentissage par simulation',
        fontsize=14, color=C['text'], fontweight='bold', y=0.95
    )

    # ----- Dessin de la grille -----
    def draw_grid(ax, snap):
        ax.clear(); ax.set_facecolor(C['panel'])
        for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        ax.set_xlim(-0.5, SIZE-0.5); ax.set_ylim(-0.5, SIZE-0.5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Politique apprise  —  episode {snap['ep']}",
                     color=C['text'], fontsize=10, pad=6)

        path_set = set(snap['trajectory'])
        pol      = snap['politique']

        for r in range(SIZE):
            for c in range(SIZE):
                s = (r, c)
                y = SIZE-1-r
                if s in env.walls:
                    fc = C['wall']
                elif s == env.goal:
                    fc = C['goal']
                elif s in path_set:
                    fc = C['path']
                else:
                    fc = C['normal']

                rect = mpatches.FancyBboxPatch(
                    (c-0.44, y-0.44), 0.88, 0.88,
                    boxstyle='round,pad=0.04',
                    facecolor=fc, edgecolor=C['bg'], lw=1.5
                )
                ax.add_patch(rect)

                if s == env.goal:
                    ax.text(c, y, '*', ha='center', va='center',
                            fontsize=18, color=C['yellow'], fontweight='bold')
                elif s in env.walls:
                    ax.text(c, y, '#', ha='center', va='center',
                            fontsize=12, color=C['muted'])
                elif s in pol:
                    ax.text(c, y, arrows[pol[s]], ha='center', va='center',
                            fontsize=17, color=C['arrow'], fontweight='bold')

        # Robot
        if snap['trajectory']:
            rr, rc = snap['trajectory'][-1]
            yr = SIZE-1-rr
            circ = plt.Circle((rc, yr), 0.28,
                               color=C['robot'], zorder=5)
            ax.add_patch(circ)
            ax.text(rc, yr, 'R', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold', zorder=6)

    # ----- Heatmap V(s) -----
    def draw_heatmap(ax, snap):
        ax.clear(); ax.set_facecolor(C['panel'])
        for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('V(s) = max Q(s,a)',
                     color=C['text'], fontsize=10, pad=6)

        V = snap['V'].copy()
        mask = np.zeros((SIZE,SIZE), dtype=bool)
        for (wr,wc) in env.walls:
            mask[wr,wc] = True
        Vm = np.ma.array(V, mask=mask)

        im = ax.imshow(Vm, cmap='RdYlGn', vmin=-15, vmax=10,
                       interpolation='nearest', aspect='equal')

        for r in range(SIZE):
            for c in range(SIZE):
                s = (r,c)
                if s in env.walls:
                    ax.text(c, r, '#', ha='center', va='center',
                            fontsize=12, color=C['muted'])
                elif s == env.goal:
                    ax.text(c, r, '*', ha='center', va='center',
                            fontsize=14, color=C['yellow'], fontweight='bold')
                else:
                    v   = V[r,c]
                    col = 'white' if v < -3 else '#111'
                    ax.text(c, r, f'{v:.1f}', ha='center', va='center',
                            fontsize=8, color=col, fontweight='bold')

        try:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color=C['muted'])
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=C['muted'], fontsize=7)
        except Exception:
            pass

    # ----- Trajectoire -----
    def draw_traj(ax, snap):
        ax.clear(); ax.set_facecolor(C['panel'])
        for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        ax.set_xlim(-0.5, SIZE-0.5); ax.set_ylim(-0.5, SIZE-0.5)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Trajectoire  —  R = {snap['reward']:.0f}",
                     color=C['text'], fontsize=10, pad=6)

        for r in range(SIZE):
            for c in range(SIZE):
                s = (r,c); y = SIZE-1-r
                fc = (C['wall'] if s in env.walls
                      else C['goal'] if s == env.goal
                      else C['normal'])
                rect = mpatches.FancyBboxPatch(
                    (c-0.44, y-0.44), 0.88, 0.88,
                    boxstyle='round,pad=0.04',
                    facecolor=fc, edgecolor=C['bg'], lw=1.5
                )
                ax.add_patch(rect)
                if s == env.goal:
                    ax.text(c, y, '*', ha='center', va='center',
                            fontsize=18, color=C['yellow'], fontweight='bold')

        traj = snap['trajectory']
        n    = len(traj)
        for i in range(n-1):
            r1,c1 = traj[i];   y1 = SIZE-1-r1
            r2,c2 = traj[i+1]; y2 = SIZE-1-r2
            alpha = 0.25 + 0.75*(i/max(n-1,1))
            col   = C['orange'] if i < n//2 else C['green']
            ax.annotate('',
                xy=(c2, y2), xytext=(c1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=col,
                    lw=1.8, alpha=alpha
                )
            )
        if traj:
            r0,c0 = traj[0]; y0 = SIZE-1-r0
            ax.plot(c0, y0, 'o', color=C['purple'], ms=9, zorder=5)

    # ----- Convergence -----
    def draw_conv(ax, snap):
        ax.clear(); ax.set_facecolor(C['panel'])
        for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        ax.tick_params(colors=C['muted'])
        ax.set_title('Courbe de convergence',
                     color=C['text'], fontsize=10, pad=6)

        hist   = snap['hist']
        smooth = snap['smooth']

        if len(hist) >= 1:
            ax.plot(np.arange(len(hist)), hist,
                    color=C['border'], lw=0.7, alpha=0.6,
                    label='Recompense brute')
        if len(smooth) >= 1:
            xs = np.arange(WINDOW-1, WINDOW-1+len(smooth))
            ax.plot(xs, smooth,
                    color=C['orange'], lw=2,
                    label=f'Moyenne ({WINDOW} ep.)')
        ax.axhline(0, color=C['muted'], lw=0.7, ls='--', alpha=0.5)
        ax.set_xlim(0, N_TOTAL)
        ax.set_xlabel('Episodes', color=C['muted'], fontsize=9)
        ax.set_ylabel('Recompense', color=C['muted'], fontsize=9)
        ax.tick_params(axis='both', colors=C['muted'])
        leg = ax.legend(facecolor=C['panel'], edgecolor=C['border'],
                        labelcolor=C['text'], fontsize=8)

    # ----- Info -----
    def draw_info(ax, snap):
        ax.clear(); ax.set_facecolor(C['panel'])
        for sp in ax.spines.values(): sp.set_edgecolor(C['border'])
        ax.axis('off')
        ax.set_title('Statistiques', color=C['text'], fontsize=10, pad=6)

        hist = snap['hist']
        moy  = np.mean(hist[-50:]) if hist else 0
        best = max(hist) if hist else 0

        # Barre de progression globale
        prog = snap['ep'] / N_TOTAL
        bar_bg = mpatches.FancyBboxPatch(
            (0.05, 0.88), 0.9, 0.06,
            boxstyle='round,pad=0.01',
            facecolor=C['normal'], edgecolor=C['border'],
            transform=ax.transAxes, clip_on=False
        )
        bar_fg = mpatches.FancyBboxPatch(
            (0.05, 0.88), 0.9*prog, 0.06,
            boxstyle='round,pad=0.01',
            facecolor=C['purple'], edgecolor='none',
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(bar_bg); ax.add_patch(bar_fg)
        ax.text(0.5, 0.91, f"{snap['ep']} / {N_TOTAL} episodes",
                ha='center', va='center', fontsize=8,
                color=C['text'], transform=ax.transAxes)

        # Stats
        rows = [
            ('Epsilon (exploration)', f"{snap['epsilon']:.4f}"),
            ('Moy recompense (50)', f"{moy:.2f}"),
            ('Meilleure recompense', f"{best:.0f}"),
            ('Etats explores',       f"{snap['n_explored']}"),
        ]
        for i, (label, val) in enumerate(rows):
            y = 0.74 - i * 0.175
            ax.text(0.05, y, label, transform=ax.transAxes,
                    fontsize=8.5, color=C['muted'], va='top')
            ax.text(0.95, y-0.02, val, transform=ax.transAxes,
                    fontsize=11, color=C['text'], va='top',
                    ha='right', fontweight='bold')
            ax.plot([0.05, 0.95], [y-0.12, y-0.12],
                    color=C['border'], lw=0.5,
                    transform=ax.transAxes, clip_on=False)

        # Barre epsilon
        ax.text(0.05, 0.06, 'Exploration', transform=ax.transAxes,
                fontsize=8, color=C['muted'])
        ep_bg = mpatches.FancyBboxPatch(
            (0.05, 0.02), 0.9, 0.05,
            boxstyle='round,pad=0.01',
            facecolor=C['normal'], edgecolor=C['border'],
            transform=ax.transAxes, clip_on=False
        )
        ep_fg = mpatches.FancyBboxPatch(
            (0.05, 0.02), 0.9*(snap['epsilon']/0.4), 0.05,
            boxstyle='round,pad=0.01',
            facecolor=C['purple'], edgecolor='none',
            transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(ep_bg); ax.add_patch(ep_fg)

    # ----- Fonction update par frame -----
    def update(frame_idx):
        snap = snapshots[frame_idx]
        draw_grid(ax_grid, snap)
        draw_heatmap(ax_heat, snap)
        draw_traj(ax_traj, snap)
        draw_conv(ax_conv, snap)
        draw_info(ax_info, snap)
        return []

    n_frames = len(snapshots)
    anim = FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=120,   # 120ms entre frames = ~8fps, fluide et lisible
        blit=False,
        repeat=False
    )

    return anim, fig


# =============================================================
# SAUVEGARDE
# =============================================================

def sauvegarder(anim, gif_path='demo_mc_control.gif',
                mp4_path='demo_mc_control.mp4'):

    # --- GIF (GitHub / LinkedIn) ---
    print(f"\nSauvegarde GIF -> {gif_path}")
    print("(patience : ~1-2 minutes selon le nombre de frames)")
    writer_gif = PillowWriter(fps=8)
    anim.save(gif_path, writer=writer_gif, dpi=90)
    print(f"GIF sauvegarde : {gif_path}")

    # --- MP4 (optionnel, necessite ffmpeg) ---
    try:
        import shutil
        if shutil.which('ffmpeg'):
            print(f"\nSauvegarde MP4 -> {mp4_path}")
            writer_mp4 = FFMpegWriter(fps=8, bitrate=1800)
            anim.save(mp4_path, writer=writer_mp4, dpi=120)
            print(f"MP4 sauvegarde : {mp4_path}")
        else:
            print("\nffmpeg non trouve -> MP4 non genere.")
            print("Pour l'activer : https://ffmpeg.org/download.html")
    except Exception as e:
        print(f"MP4 ignore ({e})")


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("DEMO LIVE — Monte Carlo Control")
    print("=" * 55)

    # 1. Pre-entrainer et capturer les snapshots
    snapshots, env = pre_entrainer(
        n_episodes   = 400,   # episodes totaux
        capture_every= 5      # 1 frame toutes les 5 episodes
    )

    # 2. Construire l'animation
    print("\nConstruction de l'animation...")
    anim, fig = construire_animation(snapshots, env)

    # 3. Sauvegarder
    sauvegarder(
        anim,
        gif_path='demo_mc_control.gif',
        mp4_path='demo_mc_control.mp4'
    )

    print("\n[OK] Termine.")
    print("Fichiers generes :")
    import os
    for f in ['demo_mc_control.gif','demo_mc_control.mp4']:
        if os.path.exists(f):
            size = os.path.getsize(f)/1024
            print(f"  {f}  ({size:.0f} Ko)")
