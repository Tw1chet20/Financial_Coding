import json
from pathlib import Path
from config import State

class StateManager:

    def __init__(self, filename='state.json'):
        self.filename = Path(filename)

    def load_state(self) -> State:

        if not self.filename.exists():
            raise FileNotFoundError(
                f'{self.filename} does not exist'
            )

        with open(self.filename, 'r') as f:
            return json.load(f)
        
    def save_state(self, state: State):

        with open(self.filename, 'w') as f:
            json.dump(state, f, indent=4)