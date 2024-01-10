from comet_ml import Experiment
from datetime import datetime

class CometLogger:
    def __init__(self, api_key, project_name):
        # Initialize the logger
        self.timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
        self.variable_steps = {}
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
        )

    def scalar(self, name, value, step=None):
        # Get the step value for the current scalar
        if step is None:
            if name in self.variable_steps:
                step = self.variable_steps[name] = self.variable_steps[name] + 1
            else:
                step = self.variable_steps[name] = 0
        else:
            self.variable_steps[name] = step

        # Log the metric
        self.experiment.log_metric(name, value, step=step)

    def flush(self):
        # End the experiment when you are done logging
        self.experiment.end()

