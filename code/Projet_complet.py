"""
CONTENU :
  Phase 1 — Fondements : demonstration numerique LGN + TCL
  Phase 2 — Implementation GridWorld + Agent Monte Carlo Control
  Phase 3 — Analyse statistique : distribution, Gelman-Rubin, ACF
  Phase 4 — Reduction de variance : Importance Sampling off-policy
  Phase 5 — Synthese visuelle finale + tableau recapitulatif

UTILISATION :
  pip install numpy matplotlib scipy
  python projet_complet.py
=============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import defaultdict

np.random.seed(42)

# =============================================================
# ==================== ENVIRONNEMENT ==========================
# =============================================================

class GridWorld:
    """
    MDP GridWorld 5x5.
    S : 25 cases (r,c), hors murs {(1,1),(1,2),(2,1)}
    A : {0=haut, 1=bas, 2=gauche, 3=droite}
    P : deterministe
    R : +10 au goal (4,4), -1 sinon
    gamma : 0.9
    """
    def __init__(self, size=5, gamma=0.9):
        self.size  = size
        self.gamma = gamma
        self.goal  = (size-1, size-1)
        self.walls = {(1,1),(1,2),(2,1)}
        self.state = None
        self.reset()

    def reset(self):
        while True:
            s = (np.random.randint(self.size), np.random.randint(self.size))
            if s != self.goal and s not in self.walls:
                self.state = s
                return s

    def step(self, action):
        moves = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
        r,c   = self.state
        dr,dc = moves[action]
        nr,nc = r+dr, c+dc
        if 0<=nr<self.size and 0<=nc<self.size and (nr,nc) not in self.walls:
            self.state = (nr,nc)
        done   = (self.state == self.goal)
        reward = 10.0 if done else -1.0
        return self.state, reward, done

    def afficher_grille(self, politique=None):
        arrows = {0:'^',1:'v',2:'<',3:'>'}
        print("\n--- GridWorld 5x5 ---")
        for r in range(self.size):
            ligne = ""
            for c in range(self.size):
                s = (r,c)
                if s == self.goal:       ligne += " [G] "
                elif s in self.walls:    ligne += " [#] "
                elif politique and s in politique: ligne += f"  {arrows[politique[s]]}  "
                else:                    ligne += "  .  "
            print(ligne)
        print()


# =============================================================
# ================ AGENT MONTE CARLO ON-POLICY ================
# =============================================================

class MonteCarloAgent:
    """
    Agent Monte Carlo Control (on-policy).
    variant : 'first_visit' ou 'every_visit'
    """
    def __init__(self, n_actions=4, gamma=0.9, epsilon=0.1, variant='first_visit'):
        self.n_actions = n_actions
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.variant   = variant
        self.Q       = defaultdict(lambda: np.zeros(n_actions))
        self.N       = defaultdict(lambda: np.zeros(n_actions))
        self.returns = defaultdict(list)
        self.hist    = []

    def choisir_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def generer_episode(self, env, max_steps=200):
        episode, state, done, steps = [], env.reset(), False, 0
        while not done and steps < max_steps:
            a = self.choisir_action(state)
            ns, r, done = env.step(a)
            episode.append((state, a, r))
            state = ns; steps += 1
        return episode

    def mettre_a_jour(self, episode):
        G, visited = 0.0, set()
        for state, action, reward in reversed(episode):
            G  = self.gamma * G + reward
            sa = (state, action)
            if self.variant == 'first_visit':
                if sa in visited: continue
                visited.add(sa)
            self.N[state][action] += 1
            n = self.N[state][action]
            self.Q[state][action] += (G - self.Q[state][action]) / n
            self.returns[sa].append(G)

    def entrainer(self, env, n_episodes=5000, decay=True):
        for i in range(n_episodes):
            ep = self.generer_episode(env)
            self.mettre_a_jour(ep)
            self.hist.append(sum(r for _,_,r in ep))
            if decay:
                self.epsilon = max(0.01, self.epsilon * 0.9995)
            if (i+1) % 1000 == 0:
                print(f"  [{self.variant}] Episode {i+1}/{n_episodes} | "
                      f"Recompense moy : {np.mean(self.hist[-100:]):.2f}")
        return self.hist

    def obtenir_politique(self, env):
        return {(r,c): int(np.argmax(self.Q[(r,c)]))
                for r in range(env.size) for c in range(env.size)
                if (r,c) != env.goal and (r,c) not in env.walls}

    def valeur_etat(self, env):
        V = np.full((env.size, env.size), np.nan)
        for r in range(env.size):
            for c in range(env.size):
                s = (r,c)
                if s == env.goal:         V[r,c] = 10.0
                elif s not in env.walls:  V[r,c] = np.max(self.Q[s])
        return V


# =============================================================
# ================ AGENT MONTE CARLO OFF-POLICY ===============
# =============================================================

def prob_action(Q_state, action, epsilon, n_actions=4):
    best = int(np.argmax(Q_state))
    return (1 - epsilon + epsilon/n_actions) if action==best else epsilon/n_actions

class OffPolicyMCAgent:
    """
    Agent Monte Carlo off-policy avec Importance Sampling.
    epsilon_pi = 0.01 (politique cible, quasi-greedy)
    epsilon_b  = 0.3  (politique comportement, exploratoire)
    """
    def __init__(self, n_actions=4, gamma=0.9, epsilon_pi=0.01, epsilon_b=0.3):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.epsilon_pi = epsilon_pi
        self.epsilon_b  = epsilon_b
        self.Q_is  = defaultdict(lambda: np.zeros(n_actions))
        self.Q_wis = defaultdict(lambda: np.zeros(n_actions))
        self.num_wis = defaultdict(lambda: np.zeros(n_actions))
        self.den_wis = defaultdict(lambda: np.zeros(n_actions))
        self.N_is    = defaultdict(lambda: np.zeros(n_actions))
        self.returns_is  = defaultdict(list)
        self.returns_wis = defaultdict(list)
        self.rhos        = defaultdict(list)
        self.hist        = []

    def action_b(self, state):
        if np.random.random() < self.epsilon_b:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q_wis[state]))

    def generer_episode_b(self, env, max_steps=200):
        episode, state, done, steps = [], env.reset(), False, 0
        while not done and steps < max_steps:
            a     = self.action_b(state)
            pb    = prob_action(self.Q_wis[state], a, self.epsilon_b, self.n_actions)
            ns, r, done = env.step(a)
            episode.append((state, a, r, pb))
            state = ns; steps += 1
        return episode

    def mettre_a_jour_is(self, episode):
        T = len(episode)
        G_vals = np.zeros(T)
        G = 0.0
        for t in range(T-1, -1, -1):
            G = self.gamma * G + episode[t][2]
            G_vals[t] = G
        rho = 1.0
        for t in range(T-1, -1, -1):
            state, action, _, prob_b = episode[t]
            pi_p = prob_action(self.Q_wis[state], action, self.epsilon_pi, self.n_actions)
            rho  = rho * (pi_p / max(prob_b, 1e-10))
            G_t  = G_vals[t]
            sa   = (state, action)
            self.N_is[state][action] += 1
            n = self.N_is[state][action]
            self.Q_is[state][action] += (rho*G_t - self.Q_is[state][action]) / n
            self.num_wis[state][action] += rho * G_t
            self.den_wis[state][action] += rho
            if self.den_wis[state][action] > 0:
                self.Q_wis[state][action] = (self.num_wis[state][action] /
                                              self.den_wis[state][action])
            self.returns_is[sa].append(rho * G_t)
            self.returns_wis[sa].append(G_t)
            self.rhos[sa].append(rho)

    def entrainer(self, env, n_episodes=5000):
        for i in range(n_episodes):
            ep = self.generer_episode_b(env)
            self.mettre_a_jour_is(ep)
            self.hist.append(sum(r for _,_,r,_ in ep))
            if (i+1) % 1000 == 0:
                print(f"  [off-policy] Episode {i+1}/{n_episodes} | "
                      f"Recompense moy : {np.mean(self.hist[-100:]):.2f}")
        return self.hist

    def valeur_etat(self, env):
        V = np.full((env.size, env.size), np.nan)
        for r in range(env.size):
            for c in range(env.size):
                s = (r,c)
                if s == env.goal:         V[r,c] = 10.0
                elif s not in env.walls:  V[r,c] = np.max(self.Q_wis[s])
        return V


# =============================================================
# ======================== PHASE 1 ============================
# =============================================================

def phase1_demo_convergence(n_sim=5000, gamma=0.9):
    print("\n[Phase 1] Demonstration numerique LGN + TCL...")
    Q_vrai = 1.0 / (1 - gamma)
    G_vals = []
    for _ in range(n_sim):
        G, t, done = 0, 0, False
        while not done:
            G   += (gamma**t) * 1.0
            t   += 1
            done = np.random.random() < (1-gamma)
        G_vals.append(G)
    G_vals  = np.array(G_vals)
    N_vals  = np.arange(1, n_sim+1)
    moyennes = np.cumsum(G_vals) / N_vals
    sigma    = G_vals.std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(N_vals, moyennes, color='#534AB7', lw=1.5, label='Q_hat(N)')
    axes[0].fill_between(N_vals,
                         moyennes - 1.96*sigma/np.sqrt(N_vals),
                         moyennes + 1.96*sigma/np.sqrt(N_vals),
                         alpha=0.2, color='#534AB7', label='IC 95%')
    axes[0].axhline(Q_vrai, color='#0F6E56', lw=1.5, ls='--',
                    label=f'Q* = {Q_vrai}')
    axes[0].set_xlabel('N episodes'); axes[0].set_ylabel('Q_hat(N)')
    axes[0].set_title('Convergence Monte Carlo (LGN)'); axes[0].legend()

    axes[1].hist(G_vals, bins=50, color='#534AB7', alpha=0.7,
                 density=True, label='G_t observes')
    x = np.linspace(G_vals.min(), G_vals.max(), 300)
    axes[1].plot(x, stats.norm.pdf(x, G_vals.mean(), G_vals.std()),
                 color='#993C1D', lw=2, label='Normale (TCL)')
    axes[1].axvline(Q_vrai, color='#0F6E56', lw=1.5, ls='--',
                    label=f'Q* = {Q_vrai:.1f}')
    axes[1].set_xlabel('G_t'); axes[1].set_ylabel('Densite')
    axes[1].set_title('Distribution des G_t (TCL)'); axes[1].legend()

    plt.suptitle('Phase 1 — LGN et TCL pour l\'estimateur Monte Carlo', fontsize=12)
    plt.tight_layout()
    plt.savefig('phase1_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Q* = {Q_vrai:.4f} | Q_hat = {moyennes[-1]:.4f} | "
          f"Biais = {abs(moyennes[-1]-Q_vrai):.6f}")
    return G_vals, Q_vrai


# =============================================================
# ======================== PHASE 2 ============================
# =============================================================

def phase2_entrainement_et_visualisation(env_fv, env_ev, agent_fv, agent_ev):
    print("\n[Phase 2] Visualisations...")

    def smooth(h, w=100):
        h = np.array(h, float)
        m = np.convolve(h, np.ones(w)/w, mode='valid')
        s = np.array([h[i:i+w].std() for i in range(len(m))])
        return m, s

    # Courbes apprentissage
    fig, ax = plt.subplots(figsize=(12, 5))
    for hist, label, color in [
        (agent_fv.hist, 'First-visit', '#534AB7'),
        (agent_ev.hist, 'Every-visit', '#0F6E56')
    ]:
        m, s = smooth(hist)
        x    = np.arange(len(m))
        ax.plot(x, m, label=label, color=color, lw=1.8)
        ax.fill_between(x, m-s, m+s, alpha=0.15, color=color)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Episodes'); ax.set_ylabel('Recompense totale (lissee)')
    ax.set_title('Phase 2 — Convergence First-visit vs Every-visit')
    ax.legend()
    plt.tight_layout()
    plt.savefig('phase2_courbes_apprentissage.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Politique et heatmap
    for agent, env, titre in [
        (agent_fv, env_fv, 'First-visit'),
        (agent_ev, env_ev, 'Every-visit')
    ]:
        arrows = {0:'^',1:'v',2:'<',3:'>'}
        V   = agent.valeur_etat(env)
        pol = agent.obtenir_politique(env)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        vmin = np.nanmin(V); vmax = np.nanmax(V)
        im = axes[0].imshow(V, cmap='RdYlGn', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[0], label='V(s)')
        for r in range(env.size):
            for c in range(env.size):
                s = (r,c)
                if s in env.walls:       axes[0].text(c,r,'#',ha='center',va='center',fontsize=12,color='white',fontweight='bold')
                elif s == env.goal:      axes[0].text(c,r,'G',ha='center',va='center',fontsize=12,color='white',fontweight='bold')
                elif not np.isnan(V[r,c]): axes[0].text(c,r,f'{V[r,c]:.1f}',ha='center',va='center',fontsize=8)
        axes[0].set_title(f'V(s) = max_a Q(s,a)\n{titre}')
        axes[1].imshow(V, cmap='RdYlGn', vmin=vmin, vmax=vmax, alpha=0.4)
        for r in range(env.size):
            for c in range(env.size):
                s = (r,c)
                if s in env.walls:   axes[1].text(c,r,'#',ha='center',va='center',fontsize=18,color='#374151')
                elif s == env.goal:  axes[1].text(c,r,'*',ha='center',va='center',fontsize=20,color='#15803D')
                elif s in pol:       axes[1].text(c,r,arrows[pol[s]],ha='center',va='center',fontsize=20,color='#1e3a5f')
        axes[1].set_title(f'Politique optimale pi*(s)\n{titre}')
        plt.suptitle(f'Phase 2 — Resultats {titre}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'phase2_politique_{titre.replace(" ","_")}.png', dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================
# ======================== PHASE 3 ============================
# =============================================================

def phase3_analyse_statistique(agent_fv, agent_ev, sa_list):
    print("\n[Phase 3] Analyse statistique...")

    # --- 3A : Distribution G_t ---
    for agent, titre in [(agent_fv,'First-visit'),(agent_ev,'Every-visit')]:
        fig = plt.figure(figsize=(14, 4*len(sa_list)))
        gs  = gridspec.GridSpec(len(sa_list), 3, figure=fig, hspace=0.55, wspace=0.35)
        for i, sa in enumerate(sa_list):
            G = np.array(agent.returns.get(sa, []))
            if len(G) < 10: continue
            mean = G.mean()
            G_t  = G[:5000] if len(G)>5000 else G
            _, p_sw = stats.shapiro(G_t)

            ax1 = fig.add_subplot(gs[i,0])
            ax1.hist(G, bins=40, density=True, color='#534AB7', alpha=0.7)
            x = np.linspace(G.min(), G.max(), 300)
            ax1.plot(x, stats.norm.pdf(x, G.mean(), G.std()), color='#993C1D', lw=2)
            ax1.axvline(mean, color='#0F6E56', lw=1.5, ls='--')
            ax1.set_title(f's={sa[0]},a={sa[1]} — Shapiro p={p_sw:.3f}', fontsize=9)
            ax1.set_xlabel('G_t'); ax1.set_ylabel('Densite')

            ax2 = fig.add_subplot(gs[i,1])
            (osm,osr),(slope,intercept,r) = stats.probplot(G, dist='norm')
            ax2.scatter(osm, osr, color='#534AB7', s=4, alpha=0.5)
            ax2.plot(osm, slope*np.array(osm)+intercept, color='#993C1D', lw=1.5)
            ax2.set_title(f'QQ-plot (R²={r**2:.4f})', fontsize=9)
            ax2.set_xlabel('Quantiles theoriques'); ax2.set_ylabel('Quantiles observes')

            ax3 = fig.add_subplot(gs[i,2])
            N_ax = np.arange(1,len(G)+1)
            cumu = np.cumsum(G)/N_ax
            sig  = G.std(ddof=1)
            ax3.plot(N_ax, cumu, color='#534AB7', lw=1.2)
            ax3.fill_between(N_ax, cumu-1.96*sig/np.sqrt(N_ax),
                             cumu+1.96*sig/np.sqrt(N_ax), alpha=0.2, color='#534AB7')
            ax3.axhline(mean, color='#0F6E56', lw=1.2, ls='--')
            ax3.set_title(f'Convergence Q({sa[0]},{sa[1]})', fontsize=9)
            ax3.set_xlabel('N'); ax3
        fig.suptitle(f'Phase 3A — Distribution G_t ({titre})', fontsize=12, y=1.01)
        plt.savefig(f'phase3_distribution_{titre.replace(" ","_")}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # --- 3B : Comparaison FV vs EV ---
    resultats = []
    print(f"\n  {'Paire':<14} {'Moy FV':>8} {'Moy EV':>8} {'Var FV':>9} {'Var EV':>9} {'p-Welch':>9}")
    for sa in sa_list:
        G_fv = np.array(agent_fv.returns.get(sa,[]))
        G_ev = np.array(agent_ev.returns.get(sa,[]))
        if len(G_fv)<5 or len(G_ev)<5: continue
        _, p_w = stats.ttest_ind(G_fv, G_ev, equal_var=False)
        print(f"  s={sa[0]},a={sa[1]:<8} {G_fv.mean():>8.3f} {G_ev.mean():>8.3f} "
              f"{G_fv.var(ddof=1):>9.3f} {G_ev.var(ddof=1):>9.3f} {p_w:>9.4f}")
        resultats.append({'sa':sa,'mean_fv':G_fv.mean(),'var_fv':G_fv.var(ddof=1),
                          'n_fv':len(G_fv),'mean_ev':G_ev.mean(),'var_ev':G_ev.var(ddof=1),
                          'n_ev':len(G_ev),'p_welch':p_w})

    labels = [f"s={r['sa'][0]},a={r['sa'][1]}" for r in resultats]
    x, w   = np.arange(len(labels)), 0.35
    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    axes[0].bar(x-w/2,[r['mean_fv'] for r in resultats],w,label='First-visit',color='#534AB7',alpha=0.85)
    axes[0].bar(x+w/2,[r['mean_ev'] for r in resultats],w,label='Every-visit',color='#0F6E56',alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels,rotation=30,fontsize=9)
    axes[0].set_title('Moyennes Q(s,a)'); axes[0].legend()
    axes[1].bar(x-w/2,[r['var_fv'] for r in resultats],w,color='#534AB7',alpha=0.85,label='First-visit')
    axes[1].bar(x+w/2,[r['var_ev'] for r in resultats],w,color='#0F6E56',alpha=0.85,label='Every-visit')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels,rotation=30,fontsize=9)
    axes[1].set_title('Variances des G_t'); axes[1].legend()
    p_vals = [r['p_welch'] for r in resultats]
    cols   = ['#0F6E56' if p>0.05 else '#993C1D' for p in p_vals]
    axes[2].bar(x, p_vals, color=cols, alpha=0.85)
    axes[2].axhline(0.05, color='black', lw=1.2, ls='--', label='Seuil 0.05')
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels,rotation=30,fontsize=9)
    axes[2].set_title('p-valeurs Test de Welch'); axes[2].legend()
    plt.suptitle('Phase 3B — Comparaison FV vs EV', fontsize=12)
    plt.tight_layout()
    plt.savefig('phase3_comparaison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- 3C : Gelman-Rubin (multi-agents) ---
    print("\n  [Gelman-Rubin] Entrainement 5 agents independants...")
    sa_gr = sa_list[0]
    agents_gr = []
    for m in range(5):
        env_m = GridWorld(size=5, gamma=0.9)
        ag    = MonteCarloAgent(variant='first_visit', gamma=0.9, epsilon=0.1)
        ag.entrainer(env_m, n_episodes=2000)
        agents_gr.append(ag)

    checkpoints = [100,200,500,1000,2000]
    R_hats, ns_gr = [], []
    for n in checkpoints:
        chaines = [np.array(ag.returns.get(sa_gr,[]))[:n]
                   for ag in agents_gr
                   if len(ag.returns.get(sa_gr,[])) >= n]
        if len(chaines) < 2: continue
        M    = len(chaines)
        N    = min(len(c) for c in chaines)
        th_j = np.array([c.mean() for c in chaines])
        th   = th_j.mean()
        B    = N/(M-1)*np.sum((th_j-th)**2)
        W    = np.mean([c.var(ddof=1) for c in chaines])
        V    = (N-1)/N*W + (M+1)/(M*N)*B
        R    = np.sqrt(V/W) if W>0 else np.inf
        R_hats.append(R); ns_gr.append(n)
        print(f"    N={n:5d} : R-hat = {R:.4f} "
              f"({'OK' if R<1.1 else 'pas converge'})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ns_gr, R_hats, 'o-', color='#534AB7', lw=2, ms=7)
    axes[0].axhline(1.1, color='#993C1D', lw=1.5, ls='--', label='Seuil 1.1')
    axes[0].axhline(1.0, color='#0F6E56', lw=1,   ls=':',  label='Convergence parfaite')
    axes[0].set_xlabel('N episodes')
    axes[0].set_ylabel('R-hat'); axes[0].set_title('Critere de Gelman-Rubin')
    axes[0].legend()
    for m, ag in enumerate(agents_gr):
        G = np.array(ag.returns.get(sa_gr,[]))
        if len(G)>0:
            cumu = np.cumsum(G)/np.arange(1,len(G)+1)
            axes[1].plot(cumu, alpha=0.7, lw=1.2, label=f'Agent {m+1}')
    axes[1].set_xlabel('Episodes'); axes[1].set_ylabel('Q_hat(N)')
    axes[1].set_title(f'Trajectoires 5 agents\ns={sa_gr[0]},a={sa_gr[1]}')
    axes[1].legend(fontsize=8)
    plt.suptitle('Phase 3C — Gelman-Rubin', fontsize=12)
    plt.tight_layout()
    plt.savefig('phase3_gelman_rubin.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- 3D : Autocorrelation ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sa_acf = sa_list[0]; max_lag = 30
    for ax, ag, label, color in [
        (axes[0], agent_fv, 'First-visit', '#534AB7'),
        (axes[1], agent_ev, 'Every-visit', '#0F6E56')
    ]:
        G = np.array(ag.returns.get(sa_acf,[]))
        if len(G) < max_lag+2: continue
        Gc  = G - G.mean()
        acf = np.correlate(Gc, Gc, mode='full')
        acf = acf[len(acf)//2:] / acf[len(acf)//2]
        lags = np.arange(max_lag+1)
        acf  = acf[:max_lag+1]
        ic_a = 1.96/np.sqrt(len(G))
        ax.bar(lags, acf, color=color, alpha=0.7)
        ax.axhline(ic_a,  color='red', lw=1, ls='--', label='IC 95%')
        ax.axhline(-ic_a, color='red', lw=1, ls='--')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f'ACF — {label}\ns={sa_acf[0]},a={sa_acf[1]}')
        ax.set_xlabel('Lag'); ax.set_ylabel('Autocorrelation')
        ax.legend(fontsize=9); ax.set_ylim(-0.3, 1.05)
    plt.suptitle('Phase 3D — Autocorrelation des G_t', fontsize=12)
    plt.tight_layout()
    plt.savefig('phase3_autocorrelation.png', dpi=150, bbox_inches='tight')
    plt.close()

    return resultats


# =============================================================
# ======================== PHASE 4 ============================
# =============================================================

def phase4_importance_sampling(agent_on, agent_off, sa_list):
    print("\n[Phase 4] Importance Sampling...")

    # Comparaison 3 estimateurs
    resultats = []
    print(f"\n  {'Paire':<14} {'On-pol':>8} {'IS-ord':>8} {'IS-pond':>8} "
          f"{'Var ON':>9} {'Var IS':>9} {'Var WIS':>9}")
    for sa in sa_list:
        G_on  = np.array(agent_on.returns.get(sa,[]))
        G_is  = np.array(agent_off.returns_is.get(sa,[]))
        rhos  = np.array(agent_off.rhos.get(sa,[]))
        G_wis = np.array(agent_off.returns_wis.get(sa,[]))
        if len(G_on)<5 or len(G_is)<5: continue
        mean_wis = (rhos*G_wis).sum()/max(rhos.sum(),1e-10)
        print(f"  s={sa[0]},a={sa[1]:<8} "
              f"{G_on.mean():>8.3f} {G_is.mean():>8.3f} {mean_wis:>8.3f} "
              f"{G_on.var(ddof=1):>9.3f} {G_is.var(ddof=1):>9.3f} {G_wis.var(ddof=1):>9.3f}")
        resultats.append({'sa':sa,'mean_on':G_on.mean(),'var_on':G_on.var(ddof=1),
                          'mean_is':G_is.mean(),'var_is':G_is.var(ddof=1),
                          'mean_wis':mean_wis,'var_wis':G_wis.var(ddof=1)})

    # Graphe comparaison
    if resultats:
        labels = [f"s={r['sa'][0]},a={r['sa'][1]}" for r in resultats]
        x, w   = np.arange(len(labels)), 0.25
        colors = ['#534AB7','#993C1D','#0F6E56']
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for i,(key,lab) in enumerate([('mean_on','On-policy FV'),
                                       ('mean_is','IS ordinaire'),
                                       ('mean_wis','IS pondere')]):
            axes[0].bar(x+(i-1)*w,[r[key] for r in resultats],w,label=lab,color=colors[i],alpha=0.85)
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels,rotation=30,fontsize=9)
        axes[0].set_title('Estimations Q(s,a)'); axes[0].legend(); axes[0].set_ylabel('Q(s,a)')
        for i,(key,lab) in enumerate([('var_on','On-policy FV'),
                                       ('var_is','IS ordinaire'),
                                       ('var_wis','IS pondere')]):
            axes[1].bar(x+(i-1)*w,[r[key] for r in resultats],w,label=lab,color=colors[i],alpha=0.85)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels,rotation=30,fontsize=9)
        axes[1].set_title('Variances des estimateurs'); axes[1].legend(); axes[1].set_ylabel('Variance')
        plt.suptitle('Phase 4 — On-policy vs IS ordinaire vs IS pondere', fontsize=12)
        plt.tight_layout()
        plt.savefig('phase4_comparaison_estimateurs.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Convergence + ratios IS
    sa_c = sa_list[0]
    G_on  = np.array(agent_on.returns.get(sa_c,[]))
    G_is  = np.array(agent_off.returns_is.get(sa_c,[]))
    rhos  = np.array(agent_off.rhos.get(sa_c,[]))
    G_wis = np.array(agent_off.returns_wis.get(sa_c,[]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for G, label, color in [
        (G_on,'On-policy FV','#534AB7'),
        (G_is,'IS ordinaire','#993C1D')
    ]:
        if len(G)<2: continue
        cumu = np.cumsum(G)/np.arange(1,len(G)+1)
        sig  = G.std(ddof=1)
        N_ax = np.arange(1,len(G)+1)
        axes[0].plot(N_ax, cumu, label=label, color=color, lw=1.5)
        axes[0].fill_between(N_ax, cumu-1.96*sig/np.sqrt(N_ax),
                             cumu+1.96*sig/np.sqrt(N_ax), alpha=0.12, color=color)
    if len(G_wis)>1:
        wis_cum = np.cumsum(rhos*G_wis)/np.maximum(np.cumsum(rhos),1e-10)
        axes[0].plot(np.arange(1,len(wis_cum)+1), wis_cum,
                     label='IS pondere', color='#0F6E56', lw=1.5)
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('Q_hat(s,a)')
    axes[0].set_title(f'Convergence s={sa_c[0]},a={sa_c[1]}'); axes[0].legend(fontsize=9)
    if len(rhos)>0:
        clip_r = np.clip(rhos, 0, np.percentile(rhos, 95))
        axes[1].hist(clip_r, bins=40, color='#993C1D', alpha=0.7, density=True)
        axes[1].axvline(rhos.mean(), color='black', lw=1.5, ls='--',
                        label=f'Moy rho={rhos.mean():.3f}')
        axes[1].axvline(1.0, color='#0F6E56', lw=1.2, ls=':', label='rho=1')
        axes[1].set_xlabel('rho = pi(a|s)/b(a|s)')
        axes[1].set_title('Distribution des ratios IS'); axes[1].legend(fontsize=9)
    plt.suptitle('Phase 4 — Convergence et ratios IS', fontsize=12)
    plt.tight_layout()
    plt.savefig('phase4_convergence_is.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Biais-variance tradeoff
    N_vals = [50,100,200,500,1000,2000]
    Q_ref  = G_on.mean() if len(G_on)>0 else 0
    b_is,v_is,b_wis,v_wis,mse_is,mse_wis,ns_bv = [],[],[],[],[],[],[]
    for N in N_vals:
        if N > len(G_is): continue
        n_boot, ei, ew = 200, [], []
        for _ in range(n_boot):
            idx = np.random.choice(len(G_is), min(N,len(G_is)), replace=True)
            ei.append(G_is[idx].mean())
            rb = rhos[idx]; gb = G_wis[idx]
            ew.append((rb*gb).sum()/max(rb.sum(),1e-10))
        ei = np.array(ei); ew = np.array(ew)
        bi = abs(ei.mean()-Q_ref); bw = abs(ew.mean()-Q_ref)
        vi = ei.var();             vw = ew.var()
        b_is.append(bi); v_is.append(vi); mse_is.append(bi**2+vi)
        b_wis.append(bw); v_wis.append(vw); mse_wis.append(bw**2+vw)
        ns_bv.append(N)
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    for ax, ys, is_, wis_, title, ylabel in [
        (axes[0], None, b_is, b_wis, 'Biais',    '|Biais|'),
        (axes[1], None, v_is, v_wis, 'Variance',  'Variance'),
        (axes[2], None, mse_is, mse_wis, 'MSE = Biais^2 + Variance', 'MSE')
    ]:
        ax.plot(ns_bv, is_,  'o-', color='#993C1D', lw=1.8, ms=6, label='IS ordinaire')
        ax.plot(ns_bv, wis_, 's-', color='#0F6E56', lw=1.8, ms=6, label='IS pondere')
        ax.set_xlabel('N')
        ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=9)
    plt.suptitle('Phase 4 — Biais-Variance Tradeoff (bootstrap 200 repetitions)', fontsize=12)
    plt.tight_layout()
    plt.savefig('phase4_biais_variance.png', dpi=150, bbox_inches='tight')
    plt.close()

    return resultats


# =============================================================
# ======================== PHASE 5 ============================
# =============================================================

def phase5_synthese(agent_fv, agent_ev, agent_off, env_fv, sa_list):
    print("\n[Phase 5] Synthese finale...")

    # --- Figure synthese : 4 panneaux en un ---
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panneau 1 : courbes apprentissage FV vs EV
    ax1 = fig.add_subplot(gs[0, :2])
    def smooth(h, w=100):
        h = np.array(h, float)
        m = np.convolve(h, np.ones(w)/w, mode='valid')
        return m
    for hist, label, color in [
        (agent_fv.hist, 'First-visit', '#534AB7'),
        (agent_ev.hist, 'Every-visit', '#0F6E56'),
        (agent_off.hist,'Off-policy IS','#993C1D')
    ]:
        m = smooth(hist)
        ax1.plot(np.arange(len(m)), m, label=label, lw=1.8)
    ax1.axhline(0, color='gray', lw=0.7, ls='--')
    ax1.set_xlabel('Episodes'); ax1.set_ylabel('Recompense lissee')
    ax1.set_title('Comparaison de convergence : 3 methodes'); ax1.legend()

    # Panneau 2 : heatmap V(s) finale (FV)
    ax2  = fig.add_subplot(gs[0, 2])
    V    = agent_fv.valeur_etat(env_fv)
    pol  = agent_fv.obtenir_politique(env_fv)
    arrows = {0:'^',1:'v',2:'<',3:'>'}
    im = ax2.imshow(V, cmap='RdYlGn', vmin=np.nanmin(V), vmax=np.nanmax(V))
    plt.colorbar(im, ax=ax2, label='V(s)')
    for r in range(env_fv.size):
        for c in range(env_fv.size):
            s = (r,c)
            if s in env_fv.walls:      ax2.text(c,r,'#',ha='center',va='center',fontsize=12,color='white')
            elif s == env_fv.goal:     ax2.text(c,r,'*',ha='center',va='center',fontsize=14,color='white',fontweight='bold')
            elif s in pol:             ax2.text(c,r,arrows[pol[s]],ha='center',va='center',fontsize=14,color='#1e3a5f')
    ax2.set_title('Politique optimale pi*(s)\n(First-visit, 5000 ep.)')

    # Panneau 3 : Tableau recapitulatif des variances
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.axis('off')
    sa_tab = [s for s in sa_list if len(agent_fv.returns.get(s,[]))>5]
    rows = []
    for sa in sa_tab:
        G_fv  = np.array(agent_fv.returns.get(sa,[]))
        G_ev  = np.array(agent_ev.returns.get(sa,[]))
        G_is  = np.array(agent_off.returns_is.get(sa,[]))
        rhos  = np.array(agent_off.rhos.get(sa,[]))
        G_wis = np.array(agent_off.returns_wis.get(sa,[]))
        if len(G_fv)<5: continue
        mwis = (rhos*G_wis).sum()/max(rhos.sum(),1e-10) if len(rhos)>0 else float('nan')
        rows.append([
            f"s={sa[0]},a={sa[1]}",
            f"{G_fv.mean():.3f}", f"{G_fv.var(ddof=1):.3f}",
            f"{G_ev.mean():.3f}", f"{G_ev.var(ddof=1):.3f}",
            f"{mwis:.3f}",        f"{G_is.var(ddof=1):.3f}"
        ])
    cols = ['Paire (s,a)',
            'Q_hat FV','Var FV',
            'Q_hat EV','Var EV',
            'Q_hat WIS','Var IS']
    table = ax3.table(cellText=rows, colLabels=cols,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for j in range(len(cols)):
        table[0,j].set_facecolor('#534AB7')
        table[0,j].set_text_props(color='white', fontweight='bold')
    ax3.set_title('Tableau recapitulatif — Estimateurs et variances', pad=20)

    # Panneau 4 : Distribution G_t premiere paire
    ax4 = fig.add_subplot(gs[1, 2])
    sa0 = sa_tab[0] if sa_tab else sa_list[0]
    for ag, label, color in [
        (agent_fv,'First-visit','#534AB7'),
        (agent_ev,'Every-visit','#0F6E56')
    ]:
        G = np.array(ag.returns.get(sa0,[]))
        if len(G)<10: continue
        ax4.hist(G, bins=35, density=True, alpha=0.5, color=color, label=label)
    ax4.set_xlabel('G_t'); ax4.set_ylabel('Densite')
    ax4.set_title(f'Distribution G_t\ns={sa0[0]},a={sa0[1]}')
    ax4.legend(fontsize=9)

    plt.suptitle('Phase 5 — Synthese : Monte Carlo Control on-policy et off-policy',
                 fontsize=13, y=1.01)
    plt.savefig('phase5_synthese.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure synthese sauvegardee : phase5_synthese.png")

    # Tableau final console
    print("\n" + "="*65)
    print("RECAPITULATIF FINAL DU PROJET")
    print("="*65)
    print(f"  {'Estimateur':<22} {'Biais':>8} {'Variance':>10} {'Convergence'}")
    print(f"  {'-'*55}")
    print(f"  {'First-visit MC':<22} {'Non biaise':>8} {'sigma^2/N':>10} {'p.s. (LGN)'}")
    print(f"  {'Every-visit MC':<22} {'Faible':>8} {'<sigma^2/N':>10} {'p.s. asymptotique'}")
    print(f"  {'IS ordinaire':<22} {'Non biaise':>8} {'peut etre grand':>10} {'p.s.'}")
    print(f"  {'IS pondere':<22} {'Biaise':>8} {'controlee':>10} {'p.s. asymptotique'}")
    print("="*65)


# =============================================================
# ========================= MAIN ==============================
# =============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("PROJET — Monte Carlo Control : Toutes les phases")
    print("=" * 65)

    # ---- PHASE 1 ----
    phase1_demo_convergence(n_sim=5000)

    # ---- PHASE 2 ----
    print("\n[Phase 2] Entrainement des agents on-policy...")
    env_fv   = GridWorld(size=5, gamma=0.9)
    agent_fv = MonteCarloAgent(variant='first_visit', epsilon=0.1)
    agent_fv.entrainer(env_fv, n_episodes=5000)

    env_ev   = GridWorld(size=5, gamma=0.9)
    agent_ev = MonteCarloAgent(variant='every_visit', epsilon=0.1)
    agent_ev.entrainer(env_ev, n_episodes=5000)

    phase2_entrainement_et_visualisation(env_fv, env_ev, agent_fv, agent_ev)

    # Paires (s,a) a analyser
    sa_list = [((0,0),3), ((0,0),1), ((2,2),1), ((3,3),1)]

    # ---- PHASE 3 ----
    phase3_analyse_statistique(agent_fv, agent_ev, sa_list)

    # ---- PHASE 4 ----
    print("\n[Phase 4] Entrainement agent off-policy IS...")
    env_off   = GridWorld(size=5, gamma=0.9)
    agent_off = OffPolicyMCAgent(gamma=0.9, epsilon_pi=0.01, epsilon_b=0.3)
    agent_off.entrainer(env_off, n_episodes=5000)

    phase4_importance_sampling(agent_fv, agent_off, sa_list)

    # ---- PHASE 5 ----
    phase5_synthese(agent_fv, agent_ev, agent_off, env_fv, sa_list)

    print("\n[OK] Projet complet termine.")
    print("Toutes les figures ont ete sauvegardees dans le dossier courant.")
    print("Figures generees :")
    import glob
    for f in sorted(glob.glob("phase*.png")):
        print(f"  {f}")
