# SmartChessV2

## Problem Statement
The aim of this project is to create some sort of algorithm to choose moves to play without having any human input to help it figure out what moves are good.  In order to do this, we must develop some catagories in our mind.

### Board State
A board state is an object that expresses all aspects of a unique board position.  It is denoted
$$
s \in S
$$
where $s$ is a board state and $S$ is the space of all possible board states.  Note that $s_0$ will be used to denote the starting position of chess.

### Move
A move is an object that describes all necessary aspects of a particular move on a chess board.  It is denoted
$$
\vec{m} \in M_s
$$
where $M_s$ the space of possible moves from state $s$ given by
$$
\gamma: S \to \{M_s\}_{s\in S},\quad \gamma(s)=M_s
$$

If a particular state in a game is denoted $s_k$, and a particular move from that state is denoted $\vec{m}_{k}$, then there exists a turn function
$$
\lambda: S \times M_{s_k} \mapsto S \quad \text{s.t.} \quad s_{k+1} = \lambda(s_k, \vec{m}_{k})
$$
The space $M_s$ is dependent on the state $s$, which is another way of saying that the possible moves from a given position depends on the position.

All the possible board states that immediately follow the current one can be expressed as
$$
F_s = \{ s_n \in S \mid s_n = \lambda(s, \vec{m}) \quad \forall \quad \vec{m} \in M_s \}
$$

### Outcome
A game outcome is denoted
$$
r \in R = \{w, l\}
$$
where $w$ is a win and $l$ is a loss (draws to be handled later...for now assume that every game ends with a win or loss).  We will, for the sake of convention, talk about the game outcome in terms of the policy about to play, $\pi_+$.  If $\pi_+$ wins, then it follows that $\pi_-$ loses.  Given some state $s$ where $\gamma(s) = \emptyset$ (i.e., the game is at a terminal point), there is a function
$$
\rho: S \mapsto R \quad \text{s.t.} \quad \rho(s) = r
$$
where $r$ is the outcome of the game at that terminal position.

### Policy
There are two ways to talk about a policy.  Normally, a policy is any function that takes a board state $s$ and returns a move $\vec{m}$.  It is denoted
$$
\pi: S \mapsto M_s \quad \text{s.t.} \quad \pi(s) = \vec{m}
$$
Such a policy is deterministic.  The space of deterministic policies is denoted $\Pi$ s.t.
$$
\pi \in \Pi_d
$$
However, in the game of chess, having deterministic policies is not that interesting, since it causes the game to go the same way each time.  Instead, lets define policies as a function that takes a state $s$ and returns a probability distribution over $M_s$
$$
\pi: S \mapsto \mathcal{P}(M_s) \\
\pi(\cdot\mid s)=\text{Categorical}\Big(P(\vec{m}_i\mid s)\Big) \\
$$
where $\vec{m}_i \in M_s$ and $P(\vec{m}_i \mid s)$ is some variable probability.  You can sample a policy to get a move in the following way.
$$
\vec{m} \sim \pi(\cdot\mid s).
$$
These policies exist in the space $\Pi$ of which $\Pi_d$ is basically a subset, since $\Pi_d$ could be thought of as all the policies that have a probability of 1 of picking any particular move.  In general, the closer that a policy is to deterministic, the more consistent and characterizable it will be.

For convinience, I will denote the policy about to move as $\pi_+$ and the policy moving next as $\pi_-$ (e.g., for $s_0$, $\pi_+$ would be playing as white and $\pi_-$ as black).

### Game
For our purposes here, envision a game as a function that takes some policies $\pi_+$ and $\pi_-$ from the set of all policies $\Pi$, as well as a board state $s$, and plays a chess game from that position until termination, returning the outcome.  It is denoted
$$
g: S \times \Pi \times \Pi \mapsto R \quad \text{s.t.} \quad g(s, \pi_+, \pi_-) = r
$$
and defined by the following recursive relationship
$$
g(s, \pi_+, \pi_-) = \begin{cases}
\rho(s), \quad \gamma(s) = \emptyset \\
g(\lambda(s, \vec{m}), \pi_-, \pi_+), \quad \gamma(s) \not = \emptyset
\end{cases} \\
$$
where
$$
\vec{m} \sim \pi_+(\cdot \mid s)
$$

### Position Quality
Let's define a space
$$
Q = [0, 1]
$$
to represent the quality of a position $s$.

If two deterministic policies, $\pi_+$ and $\pi_-$ play a game starting from a given starting state $s$, then the probability of winning will always be 1 or 0,
$$
P(g(s, \pi_+, \pi_-) = w) \in \{ 0, 1 \}
$$

since chess itself is deterministic.  However, if we instead play the game with non-deterministic policies $\pi'_+$ and $\pi'_-$, then we find that
$$
P(g(s, \pi'_+, \pi'_-) = w) \in Q
$$
To save notation space, I'm going to define a new function $p$ that is defined in the following way,
$$
p: S \times \Pi \times \Pi \mapsto Q \\
\text{s.t.} \quad p(s, \pi_+, \pi_-) = P(g(s, \pi_+, \pi_-)=w)
$$
Now, if both policies are reasonably good at playing chess (more on this later), $p(s, \pi_+, \pi_-)$ is a reasonablly good estimation of the quality of the board state $s$ for $\pi_+$ and $1 - p(s, \pi_+, \pi_-)$ for $\pi_-$.

What does it actually mean to be "good at chess", though?  We can intuitively understand that there are some people better than others at playing chess, but is there any objective best play from which we can assess the universal value of a given position?It turns out, there is! First, we must establish some concepts.

Against any particular policy, $\pi$, there exists a foil policy, $\tilde{\pi}$, satisfying all the requirements of a policy with the defining characteristic
$$
\tilde{\pi}(s) = \arg \min_{\vec{m} \in M_{s}} p(\lambda(s, \vec{m}), \pi, \tilde{\pi})
$$
stating that $\tilde{\pi}$ picks the move that minimizes the probability of $\pi$ winning.

Now, optimal play can be defined by the policy $\pi_{\infty}$, which is defined with the expression
$$
\pi_{\infty}(s) = \arg \min_{\vec{m} \in M_{s}} p(\lambda(s, \vec{m}), \tilde{\pi}_{\infty}, \pi_{\infty})
$$
which essentially states the $\pi_{\infty}$ plays in such a way to minimize the effectiveness of its own foil at beating it.  Note that this entails that $\pi_{\infty}$ and $\tilde{\pi}_{\infty}$ must be the same policy, since they are defined the same way.

So, given that $\pi_{\infty}$ is the policy that plays optimally, the absolute quality (for the policy about to move) of a given position $s$ could be expressed as
$$
q_s = p(s, \pi_{\infty}, \pi_{\infty})
$$
where $q_s \in Q$.

If given any two policies $\pi_i$ and $\pi_j$ in $\Pi$,
$$
p(s, \pi_i, \pi_j) \not = p(s, \pi_{\infty}, \pi_{\infty})
$$
but it will approach it as $\pi_i, \pi_j \to \pi_{\infty}$.

### Model
A model is an object $\mu$ that is defined by the following
$$
\mu: S \mapsto Q \quad \text{s.t.} \quad \mu(s) = q
$$
where $q \in Q$.  Models can be used to define a *deterministic* policy in the following way:
$$
\pi(s) = \arg\min_{\vec{m}\in M_s} \mu\big(\lambda(s,\vec{m})\big)
$$
Note that this is minimizing because the model will, by default, evaluate from the perspective of the player about to play.

The notion of a model can also help us define a *non-deterministic* policy, $\pi$.  For any move $\vec{m}_i$, the quality of the position after the move for the policy that moved is given by
$$
q_i = 1 - \mu(\lambda(s, \vec{m}_i))
$$
then we can define the probability of a particular move being chosen in terms of the softmax function:
$$
P(\vec{m}_i \mid s) = \frac{\exp(\beta, q_i)}{\sum_j \exp(\beta, q_j)},\quad \beta=\frac{1}{T}
$$
where $T$ is the tempreture of the softmax.  Large $T$ vallues will yeild a largely uniform distribution over all the possible moves, while small $T$ values will focus the distribution on moves that the model believes are promising.  Using this probability measure, we can define
$$
\pi: S \to \mathcal{P}(M_s) \\
\pi(\cdot\mid s)=\{ P(\vec{m}_0 \mid s), P(\vec{m}_1 \mid s), P(\vec{m}_2 \mid s), \: \cdots, P(\vec{m}_{N-1} \mid s) \}
$$
where $\vec{m} \sim \pi(s)$ is the move sampled from $\pi$ at state $s$.

Observe how, if $\mu(s) = p(s, \pi_{\infty}, \pi_{\infty})$, the policy $\pi$ defined by $\mu$ will approach $\pi_{\infty}$ as $T \to 0$.  Knowing this, we can see that in order to play good chess, we must be able to build a model $\mu$ such that
$$
\mu(s) \approx p(s, \pi_{\infty}, \pi_{\infty})
$$

### Game Trajectory
A game trajectory is just the ordered set of game states
$$
\vec{s} = (s_0, s_1, s_2, \cdots, s_{N-1})
$$
where $s_{N-1}$ is the terminal state of the game ( $\gamma(s_{N-1}) = \emptyset$ ).  The length of $\tau$ is $N$ and the policy to move starts with the one playing white and alternates between white and black for the whole game.  A game trajectory can be used to evaluate the behavior of a particular model over the course of a game, and give us insights into how to train the model.

Take the model evalation vector $\vec{p}_{white}$ and its compliment $\vec{p}_{black}$ defined by
$$
p_{i \mid \text{white}} = \begin{cases}
\mu(s_i), & i \equiv 0 \pmod 2\\[4pt]
1-\mu(s_i), & i \equiv 1 \pmod 2
\end{cases} \\
p_{i \mid \text{black}} = 1 - p_{i \mid \text{white}}
$$
where
$$
\mu(s_{N-1}) = \begin{cases}
1, \quad \rho(s_{N-1}) = w \\
0, \quad \rho(s_{N-1}) = l
\end{cases}
$$
In a well trained model, this quantity begins close to 0.5 and makes its way either to 1 or 0 as the game goes on (1 if the policy is winning, or 0 if losing).

![Normal Trajectory](graphs/Example%20Game%20Trajectory%20Board%20Eval)

### Model Reinforcement
The challenge now is to decide how to reinforce a model $\mu(s)$ so that it becomes a better estimate of $p(s, \pi_{\infty}, \pi_{\infty})$.  Say I have some policy $\pi$ defined by the model $\mu$.  If I transform $\mu$ into $\mu'$, by the following rule
$$
\mu'(s) = p(s, \pi, \pi)
$$
it is taken as given that the following is true about the model $\pi'$ parameterized by $\mu'$:
$$
p(s_0, \pi', \pi) > 0.5 \\
p(s_0, \pi, \pi') > 0.5
$$
This postulate is taken as a given, and is not proven here.  I believe that it is sufficiently intuitive and it has been backed up by exprimental evience in the [first chess bot](https://github.com/eRaquet/SmartChess).  This assumption means that if we simply find a way to estimate $p(s, \pi, \pi)$, we can improve $\pi$.

Along with this assumption, we also assume that if
$$
p(s_0, \pi', \pi) > 0.5 \\
p(s_0, \pi, \pi') > 0.5
$$
and
$$
p(s_0, \pi'', \pi') > 0.5 \\
p(s_0, \pi', \pi'') > 0.5
$$
then
$$
p(s_0, \pi'', \pi) > 0.5 \\
p(s_0, \pi, \pi'') > 0.5
$$
This to is intuitive, but is not necessarily entirely true.  We will address potential issues with this assumption later.

The most intuitive way to estimate $p(s, \pi, \pi)$ is to play a bunch of games between two instances of $\pi$ and manually calculate the probability.  In the RL context, this would look like applying an equal positive reinforcement to all the game states if the game was won and a equal negative reinforcement to all the game states if lost.  While this seems promising, it actually leads to very bad training results.  The main issue is that the size of $S$ is so absurdly large that the chance of ever seeing a mid/late game position twice is virtually assured to be null.

In addition to this downside, actual experience shows that models trained this way are very noisy, often having virtually no consistency between positions.  Their evaluation trajectories look more like

![Uncorrelated Trajectory](graphs/Example%20Game%20Trajectory%20Board%20Eval%20Uncorrelated)

We can clearly see that this is very wrong, as the evaluation of a particular board state is intimately tied to the evaluations of the surrounding states.  Worse, this reinforcement tends to saturate the evaluations early, even if the policy loses after suggesting a win confidence of 1.

![Saturated Trajectory](graphs/Example%20Game%20Trajectory%20Board%20Eval%20Saturated)

There is a better way...  Consider game trajectory board evaluation plot for white:

![Blunder Trajectory](graphs/Example%20Game%20Trajectory%20Board%20Eval%20Blunder)

This pictures a blunder.  Under the first reinforcement criteria, all the moves from 0 to 44 should be negatively reinforced, since the policy ended up losing.  However, this is obviously the wrong thing to do, since it is clear from the trajectory that the policy lost **only** because of move 45.  Essentially, the policy picked a state that it thought was good, but then the next state revealed (likely by the opponent's response) that the chosen state was actually very bad.  And here's the real clincher...how bad was the chosen state?  Approximately as bad as the state that followed!  After all, if you assume that the opponent is trying to pick the state that minimizes your chance of winning, then the quality of any state you choose is only as good as the quality (for you) of the state your opponent chooses as a response.

This insight allows us to propose an alternate reinforcement expressed
$$
\mu(s_k) = \frac{\mu(s_k) + \alpha(1 - \mu(s_{k+1}))}{1 + \alpha}
$$
where $\alpha \in [0, \infty]$ is the reinforcement strength factor.  There are a number of advantages to this reinforcement:
- It enforces consistency between both players, ensuring that if a position is good for white it is also equally bad for black.
- It also enforces consistency with the game trajectory itself, ensuring that related board states have related quality.
- It is a balanced reinforcement, ensuring that the absolute amount of positive and negative reinforcement is equal to each other.
- It is particularly good at identifying and correcting specific moves that cause issues.
- It is also good at taking advantages of opponent weakness when possible.

I believe this reinforcement is close (if not by definition exactly) the optimal way to take the game outcome and reflect it upon the rest of the game states.