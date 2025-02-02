from datetime import datetime
import os
import numpy as np

class MLExperimentLogger:
    def __init__(self, log_file="ml_experiments.md"):
        """Initialize the experiment logger with a markdown file."""
        self.log_file = log_file
        
        # Create file with header if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write("# Machine Learning Experiments Log\n\n")
    
    def log_experiment(self, cv_scores=None, test_metrics=None):
        """Log a new experiment with single-line inputs."""
        # Get experiment details through simple prompts
        exp_details = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": input("Model name: "),
            "changes": input("Changes made: "),
            "reasoning": input("Reasoning behind changes: "),
            "hyperparameters": self._get_hyperparameters(),
            "results": self._get_results(cv_scores, test_metrics),
            "observations": input("Observations from this experiment: ")
        }
        
        # Create markdown content
        markdown_content = self._format_experiment(exp_details)
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(markdown_content)
        
        print(f"\nExperiment logged successfully in {self.log_file}!")
    
    def _get_hyperparameters(self):
        """Get hyperparameters as a single string."""
        params_str = input("Hyperparameters (format: param1=value1, param2=value2): ")
        params = {}
        
        if params_str:
            for param in params_str.split(','):
                try:
                    key, value = param.strip().split('=')
                    # Try to convert value to numeric if possible
                    try:
                        value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass
                    params[key.strip()] = value
                except ValueError:
                    print(f"Warning: Skipping invalid parameter format: {param}")
        
        return params
    
    def _get_results(self, cv_scores=None, test_metrics=None):
        """Process and format results."""
        results = {}
        
        if cv_scores is not None and len(cv_scores) > 0:
            # Convert to list if numpy array
            cv_scores_list = cv_scores.tolist() if isinstance(cv_scores, np.ndarray) else cv_scores
            results["cv_scores"] = cv_scores_list
            results["mean_cv_score"] = float(np.mean(cv_scores))
            results["cv_std"] = float(np.std(cv_scores))
        
        if test_metrics:
            results["test_metrics"] = test_metrics
        
        return results
    
    def _format_experiment(self, exp):
        """Format experiment details as markdown."""
        markdown = f"""
## Experiment - {exp['datetime']}

### Model
{exp['model']}

### Changes Made
{exp['changes']}

### Reasoning
{exp['reasoning']}

### Hyperparameters
```python
{self._format_dict(exp['hyperparameters'])}
```

### Results
{self._format_results(exp['results'])}

### Observations
{exp['observations']}

---
"""
        return markdown
    
    def _format_dict(self, d):
        """Format dictionary in a readable way."""
        return '\n'.join(f"{k}: {v}" for k, v in d.items()) if d else "None"
    
    def _format_results(self, results):
        """Format all results."""
        output = ""
        
        # CV Scores
        if "cv_scores" in results:
            output += "#### Cross-Validation Scores\n"
            output += "Individual scores:\n"
            for i, score in enumerate(results['cv_scores'], 1):
                output += f"- Fold {i}: {score:.4f}\n"
            
            output += f"\nSummary:\n"
            output += f"- Mean: {results['mean_cv_score']:.4f}\n"
            output += f"- Std: {results['cv_std']:.4f}\n\n"
        
        # Test Metrics
        if "test_metrics" in results:
            output += "#### Test Metrics\n"
            for metric, value in results['test_metrics'].items():
                output += f"- {metric}: {value:.4f}\n"
        
        return output if output else "No results recorded"