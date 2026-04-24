import os
import time
import torch
import numpy as np
from model import ConnectFourNet
from genetics import next_generation, score_population, create_population
import json 

from multiprocessing import Pool, cpu_count
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# --- Config ---
POPULATION_SIZE = 100
N_GENERATIONS = 200
ELITE_FRAC = 0.2
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.1
GAMES_PER_OPPONENT = 6
SAVE_DIR = "checkpoints"
SAVE_EVERY = 10  # Save best model every N generations
N_WORKERS = max(1, cpu_count() - 1)  # Leave one core free
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Multiprocessing ---

def score_one(args):
    """
    Worker function — scores a single model against a sample of opponents.
    Must be a top-level function for multiprocessing to pickle it.
    """
    model_weights, opponent_weights_list, games_per_opponent = args

    device = torch.device('cpu')

    model = ConnectFourNet(device=device)
    model.set_weights(model_weights)

    opponents = []
    for w in opponent_weights_list:
        opp = ConnectFourNet(device=device)
        opp.set_weights(w)
        opponents.append(opp)

    from game_runner import evaluate
    return evaluate(model, opponents, games_per_opponent)


def score_population_parallel(population, games_per_opponent=6, n_workers=N_WORKERS):
    """
    Scores the entire population in parallel.
    Converts models to weight arrays for pickling, then reconstructs in workers.
    """
    all_weights = [m.get_weights() for m in population]
    sample_size = min(10, len(population) - 1)

    args_list = []
    for i in range(len(population)):
        opponent_indices = [j for j in range(len(population)) if j != i]
        sampled = np.random.choice(opponent_indices, size=sample_size, replace=False)
        opponent_weights = [all_weights[j] for j in sampled]
        args_list.append((all_weights[i], opponent_weights, games_per_opponent))

    with Pool(processes=n_workers) as pool:
        scores = pool.map(score_one, args_list)

    scored = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    return scored


# --- Checkpointing ---

def save_model(model, generation, score):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"best_gen_{generation:04d}_score_{score:.3f}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Saved checkpoint: {path}")


def load_model(path):
    model = ConnectFourNet()
    model.load_state_dict(torch.load(path))
    return model


# --- Training Loop ---

def train(resume_path=None):
    print(f"Starting training — {POPULATION_SIZE} models, {N_GENERATIONS} generations")
    print(f"Using {N_WORKERS} workers")
    print(f"Device: {DEVICE}\n")

    # Initialise or resume
    if resume_path:
        print(f"Resuming from {resume_path}")
        seed_model = load_model(resume_path)
        # Seed the population from the checkpoint with mutations
        population = [ConnectFourNet() for _ in range(POPULATION_SIZE)]
        seed_weights = seed_model.get_weights()
        for m in population:
            noise = np.random.randn(len(seed_weights)) * 0.1
            m.set_weights(seed_weights + noise)
    else:
        population = create_population(POPULATION_SIZE)

    best_ever_score = float('-inf')
    best_ever_model = None

    for gen in range(1, N_GENERATIONS + 1):
        start = time.time()

        # Score
        scored = score_population_parallel(population, GAMES_PER_OPPONENT, N_WORKERS)
        best_score = scored[0][0]
        avg_score = sum(s for s, _ in scored) / len(scored)
        best_model = scored[0][1]

        # Track best ever
        if best_score > best_ever_score:
            best_ever_score = best_score
            best_ever_model = best_model

        # Save checkpoint
        if gen % SAVE_EVERY == 0:
            save_model(best_model, gen, best_score)

        elapsed = time.time() - start
        print(f"Gen {gen:4d} | Best: {best_score:+.3f} | Avg: {avg_score:+.3f} | "
              f"Best Ever: {best_ever_score:+.3f} | Time: {elapsed:.1f}s")

        # Breed next generation
        n_elites = max(2, int(POPULATION_SIZE * ELITE_FRAC))
        elites = [m for _, m in scored[:n_elites]]

        from genetics import breed_new_generation
        population = breed_new_generation(
            elites,
            POPULATION_SIZE,
            MUTATION_RATE,
            MUTATION_STRENGTH
        )
        all_scores = [s for s, _ in scored]
        write_dashboard_log(gen, best_score, avg_score, elapsed, all_scores)

    # Final save
    save_model(best_ever_model, N_GENERATIONS, best_ever_score)
    print(f"\nTraining complete. Best score: {best_ever_score:+.3f}")
    return best_ever_model

    
def write_dashboard_log(gen, best_score, avg_score, elapsed, population_scores):
    entry = {
        "gen": gen,
        "best": round(float(best_score), 4),
        "avg": round(float(avg_score), 4),
        "time": round(elapsed, 2),
        # Full score distribution for the population histogram
        "dist": [round(float(s), 3) for s in sorted(population_scores)]
    }
    with open("training_log.json", "a") as f:
        f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(resume_path=args.resume)
