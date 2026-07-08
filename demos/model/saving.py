"""Script to demo the StandardModel class saving mechanism."""

from modules.model import StandardModel

if __name__ == "__main__":
    model = StandardModel(0, 0)
    model.save(keep_generation=True)
    model.save(new_generation=True)
