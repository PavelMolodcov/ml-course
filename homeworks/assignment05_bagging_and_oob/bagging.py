import numpy as np

from sklearn.metrics import mean_squared_error
class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
    
    def _generate_splits(self, data: np.ndarray):
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, data_length, replace=True)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        self.data = data
        self.target = target
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain len(data) number of elements!'
        self.models_list = []
        for indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[indices], target[indices]
            self.models_list.append(model.fit(data_bag, target_bag))
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        predictions = np.zeros((len(data), len(self.models_list)))
        for i, model in enumerate(self.models_list):
            predictions[:, i] = model.predict(data)
        return np.mean(predictions, axis=1)
    
    def _get_oob_predictions_from_every_model(self):
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for i, (data_point, target_point) in enumerate(zip(self.data, self.target)):
            predictions = []
            for j, indices in enumerate(self.indices_list):
                if i not in indices:
                    predictions.append(self.models_list[j].predict([data_point])[0])
            list_of_predictions_lists[i] = predictions
        self.list_of_predictions_lists = list_of_predictions_lists

    def _get_averaged_oob_predictions(self):
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = [np.mean(pred) if pred else None for pred in self.list_of_predictions_lists]


    def OOB_score(self):
        self._get_averaged_oob_predictions()
        mask = [pred is not None for pred in self.oob_predictions]
        oob_data = self.data[mask]
        oob_target = self.target[mask]
        oob_predictions = np.array([pred for pred in self.oob_predictions if pred is not None])
        return mean_squared_error(oob_target, oob_predictions)