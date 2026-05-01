"""Script to seed all the model strains when beginning a training session."""

from modules.model import StandardModel

if __name__ == "__main__":
    for i in range(8):
        StandardModel(i, 0, construct=True)
