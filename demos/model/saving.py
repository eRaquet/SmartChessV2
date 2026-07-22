"""Script to demo the StandardModel class saving mechanism."""

from smartchess.model import InferenceModel

if __name__ == '__main__':
    model = InferenceModel(0, 0)
    model.save(keep_generation=True)
    model.save(new_generation=True)
