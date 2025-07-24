import random
import itertools
import yaml
import json


class OfflineSweep:
    """
    A class to perform offline hyperparameter sweeps.
    """

    def __init__(self, sweep_config, method=None, seed=None, unique=False):
        """
        Initialize the OfflineSweep with a sweep configuration.
        :param sweep_config: A dictionary or a file path to a JSON/YAML file containing the sweep configuration.
        :param method: The method of the sweep, either "random" or "grid". If provided, this takes precedence over the config's "method" key.
        :param seed: Random seed for reproducibility.
        :param unique: Whether to sample unique parameter combinations (only applicable for random method).
        """
        # Accept a dict directly instead of a file path
        if isinstance(sweep_config, str):
            # If a string is passed, treat it as a file path for backward compatibility
            with open(sweep_config, "r") as f:
                if sweep_config.endswith(".json"):
                    self.sweep_config = json.load(f)
                elif sweep_config.endswith(".yaml") or sweep_config.endswith(".yml"):
                    self.sweep_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format {sweep_config}. Use JSON or YAML or a dict directly.")
        elif isinstance(sweep_config, dict):
            self.sweep_config = sweep_config
        else:
            raise ValueError("sweep_config must be a dictionary or a file path to a JSON/YAML file.")
        self.parameters = self.sweep_config.get("parameters", {})

        if seed is None:
            self._rng = random.Random()
        else:
            self._rng = random.Random(seed)
        # Use method argument if provided, otherwise fall back to config's "method" key
        self.method = method if method is not None else self.sweep_config.get("method", "grid")
        self.unique = unique
        self._unique_samples = set()

    def _get_param_values(self, param):
        conf = self.parameters[param]
        if "values" in conf:
            return conf["values"]
        elif "min" in conf and "max" in conf:
            if conf.get("distribution", "uniform") == "uniform":
                return [self._rng.uniform(conf["min"], conf["max"])]
            elif conf.get("distribution") == "int_uniform":
                return [self._rng.randint(conf["min"], conf["max"])]
        elif "value" in conf:
            return [conf["value"]]
        return []

    def _random_sample(self):
        sample = {}
        for param in self.parameters:
            values = self._get_param_values(param)
            if len(values) == 1:
                sample[param] = values[0]
            else:
                sample[param] = self._rng.choice(values)
        return sample

    def _max_samples(self):
        """
        Calculate the maximum number of samples based on the sweep configuration.
        :return: The maximum number of unique samples that can be generated.
        """
        max_samples = 1
        for param in self.parameters:
            if "values" in self.parameters[param]:
                max_samples *= len(self.parameters[param]["values"])
            elif "value" in self.parameters[param]:
                max_samples *= 1
            elif "min" in self.parameters[param] and "max" in self.parameters[param]:
                return -1  # Infinite samples for continuous ranges
        return max_samples

    def _random_samples(self, num_samples):
        # Ensure num_samples of unique samples can be generated
        max_samples = self._max_samples()
        if max_samples != -1 and num_samples > max_samples:
            raise ValueError(
                f"Requested {num_samples} samples, but only {max_samples} unique samples can be generated."
            )
        # Generate random samples
        self._unique_samples.clear()  # Clear unique samples set
        count = 0
        while num_samples is None or count < num_samples:
            sample = self._random_sample()
            if self.unique:
                # Use tuple of sorted items for uniqueness
                sample_tuple = tuple(sorted(sample.items()))
                if sample_tuple in self._unique_samples:
                    continue
                self._unique_samples.add(sample_tuple)
            yield sample
            count += 1

    def _grid_samples(self):
        keys = list(self.parameters.keys())
        value_lists = [self._get_param_values(k) for k in keys]
        for values in itertools.product(*value_lists):
            yield dict(zip(keys, values))

    def samples(self, num_samples=None):
        """
        Generate samples based on the sweep configuration.
        :param num_samples: Number of samples to generate. If None, all combinations will be generated.
        :return: A generator yielding parameter dictionaries.
        """
        if self.method == "grid":
            return list(self._grid_samples())
        elif self.method == "random":
            return list(self._random_samples(num_samples))
        else:
            raise ValueError(f"Unsupported method: {self.method}")
