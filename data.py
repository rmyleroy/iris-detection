import enum
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Direction(enum.Enum):
    """Enum class discribing the different possible directions"""
    UNDEFINED = 1
    CENTER = 2
    UP = 3
    LEFT = 4
    RIGHT = 5
    DOWN = 6

    @classmethod
    def getString(cls, direction):
        """Get the text corresponding to a given direction

        Args:
            direction (Direction): Requested direction

        Returns:
            str: Direction name
        """
        return {
            cls.UNDEFINED: '?',
            cls.CENTER: 'center',
            cls.UP: 'up',
            cls.LEFT: 'left',
            cls.RIGHT: 'right',
            cls.DOWN: 'down',
        }[direction]

    @classmethod
    def successor(cls, direction):
        """Next direction in enum

        Args:
            direction (Direction): Current direction

        Returns:
            Direction: Next direction in enum
        """
        if direction == cls.DOWN:
            return cls.CENTER
        return Direction(direction.value + 1)

    @classmethod
    def precessor(cls, direction):
        """Previous direction in enum

        Args:
            direction (Direction): Current direction

        Returns:
            Direction: Previous direction in enum
        """
        if direction == cls.CENTER:
            return cls.DOWN
        return Direction(direction.value - 1)


class Data(object):
    """Entry in the dataset, representing eyes, their moments and the eye direction

    Args:
        frame (np.array): Original video frame
        left_eye (Eye): Left eye detected
        right_eye (Eye): Right eye detected
        direction (Direction): Direction label

    Attributes:
        left_moments (type): Vector moments of the left eye
        right_moments (type): Vector moments of the right eye
        frame
        left_eye
        right_eye
        direction
    """
    def __init__(self, frame, left_eye, right_eye, direction=Direction.UNDEFINED):
        self.frame = np.copy(frame)
        self.left_eye = left_eye.__dict__
        self.right_eye = right_eye.__dict__
        self.direction = direction.value

        self.left_moments = left_eye.computeMomentVectors()
        self.right_moments = right_eye.computeMomentVectors()


class Dataset(object):
    """A Dataset in which data will be stored and used to estimate a look direction

    Attributes:
        NB_NEIGHBOURS (int): Number of neighbours considered in the nearest neighbour selection
        PERCENT_TRAINING_SET (float): Part of the dataset that should be used as a training set (and the rest should go to the validation set)
        NUMBER_MEAN_SCORE (int): number of cross validation test done
    """
    NB_NEIGHBOURS = 3
    PERCENT_TRAINING_SET = 0.7
    NUMBER_MEAN_SCORE = 20

    def __init__(self):
        self.dataset_file = "dataset"
        self.clear()

    def append(self, data):
        """Adds an element to the dataset
        """
        if data not in self.data:
            self.data.append(data)

    def deleteLastEntry(self):
        """Deletes the lastly added element from the dataset

        Returns:
            bool: True if an element has been deleted
        """
        if self.data:
            self.data = self.data[:-1]
            return True
        return False

    def clear(self):
        """Empties the dataset
        """
        self.data = []

    def __len__(self):
        return len(self.data)

    def save(self):
        """Save the dataset into a file
        """
        pickle.dump(self.data, open(self.dataset_file, 'wb'))

    def load(self):
        """Load a saved dataset from a file
        """
        try:
            self.data = pickle.load(open(self.dataset_file, 'rb'))
        except Exception as e:
            print(e)

    def leftMoments(self, ids=None):
        """Retrieves an array of vector moments for the left eyes from the stored data.
        Returned array shape is (nb elements * 7)

        Args:
            ids (np.array): Only retrieves data from the given indices

        Returns:
            np.array: Description of returned object.
        """
        data = self.data if ids is None else np.array(self.data)[ids]
        if len(data) == 0:
            return []
        return np.stack([d.left_moments for d in data])

    def rightMoments(self, ids=None):
        """Retrieves an array of vector moments for the right eyes from the stored data.
        Returned array shape is (nb elements * 7)

        Args:
            ids (np.array): Only retrieves data from the given indices

        Returns:
            np.array: Description of returned object.
        """
        data = self.data if ids is None else np.array(self.data)[ids]
        if len(data) == 0:
            return []
        return np.stack([d.right_moments for d in data])

    def labels(self, ids=None):
        """Retrieves an array of vector moments for the direction labels from the stored data.
        Returned array is a 1d array of shape (nb elements)

        Args:
            ids (np.array): Only retrieves data from the given indices

        Returns:
            np.array: Description of returned object.
        """
        data = self.data if ids is None else np.array(self.data)[ids]
        if len(data) == 0:
            return []
        return np.array([d.direction for d in data])

    def directionProbabilities(self, moment_left, moment_right, idsTraining=None):
        """Estimates the look direction probabilities for each possible direction, accordingly to eyes moments.
        We use a k-nearest neighbour to identify closest stored moments to the current moments and chose the most
        represented direction among this neighbours.

        Args:
            moment_left (np.array): Vector moments of the left eye (1d array of size 7)
            moment_right (np.array): Vector moments of the right eye (1d array of size 7)
            idsTraining (np.array): Indices of the data stored in the dataset that should be used for this estimation

        Returns:
            np.array: Array of tuples, each element is (Direction, direction probability)
        """
        if not self.data:
            return []
        labels = self.labels(idsTraining)

        all_distances_left = np.sum(np.power(moment_left - self.leftMoments(idsTraining), 2), 1)
        best_args_left = labels[np.argsort(all_distances_left)[:self.NB_NEIGHBOURS]] - 1
        scores_left = np.histogram(best_args_left, bins=np.arange(len(Direction) + 1))[0]

        all_distances_right = np.sum(np.power(moment_right - self.rightMoments(idsTraining), 2), 1)
        best_args_right = labels[np.argsort(all_distances_right)[:self.NB_NEIGHBOURS]] - 1
        scores_right = np.histogram(best_args_right, bins=np.arange(len(Direction) + 1))[0]

        return np.array(list(zip(Direction, np.sum([scores_left, scores_right], 0))))

    def estimateDirection(self, moment_left, moment_right, idsTraining=None):
        """Estimates the eye direction, given the eye vector moments.

        Args:
            moment_left (np.array): Vector moments of the left eye (1d array of size 7)
            moment_right (np.array): Vector moments of the right eye (1d array of size 7)
            idsTraining (np.array): Indices of the data stored in the dataset that should be used for this estimation

        Returns:
            Direction: Estimated direction
        """
        best_direction = Direction.UNDEFINED
        scores = self.directionProbabilities(moment_left, moment_right, idsTraining=idsTraining)
        if len(scores) > 0:
            best_direction = scores[np.argmax(scores[:, 1]), 0]
        return best_direction

    def getValidationScore(self, maxLimit=None):
        """Performs a cross validation evaluation of the eye estimation.

        Args:
            maxLimit (int): The n-first elements of the dataset to be taken in account for this score evaluation.

        Returns:
            float: Score evaluation (success rate)
        """
        globalScores = []
        for i in range(self.NUMBER_MEAN_SCORE):
            idsTraining, idsValidation = self.getCrossValidationIds(maxLimit)
            scores = []
            for left_moments, right_moments, label in zip(self.leftMoments(idsValidation),
                                                          self.rightMoments(idsValidation),
                                                          self.labels(idsValidation)):
                estimation = self.estimateDirection(left_moments, right_moments, idsTraining=idsTraining)
                scores.append(float(label == estimation.value))
            globalScores.append(np.mean(scores))
        score = np.mean(globalScores)
        return score

    def getValidationScoreEvolution(self, step=1):
        """Computes the scores obtained when successively considering the 2, 3, ... k-first elemnt of the dataset.
        This helps to evaluate the evolution of the score.

        Args:
            step (int): step to increase the number of elements used by

        Returns:
            list: List of tuples, each one is (number of elements used, score)
        """
        scores = []
        for i in range(2, len(self.data), step):
            scores.append((i, self.getValidationScore(i)))
        return scores

    def getCrossValidationIds(self, maxLimit=None):
        """Creates a pair of training - validation sets.

        Args:
            maxLimit (int): The n-first elements of the dataset to be considered.

        Returns:
            (np.array, np.array): indices of the elements of the training set and of the validation set
        """
        maxLimit = maxLimit if maxLimit else len(self.data)
        numberTraining = max(1, int(maxLimit * self.PERCENT_TRAINING_SET))
        ids = np.arange(maxLimit)
        np.random.shuffle(ids)
        idsValidation = ids[numberTraining:]
        idsTraining = ids[:numberTraining]
        return idsTraining, idsValidation

    def drawVectorizedMoments(self):
        """Show a matrix plot of the vector moments stored in the dataset.

        Returns:
            plt.Figure: Figure created
        """
        x_data_left = np.copy(self.leftMoments())
        x_data_right = np.copy(self.rightMoments())
        y_data = np.copy(self.labels())
        colors = {
            Direction.CENTER: 'k',
            Direction.UP: 'r',
            Direction.LEFT: 'g',
            Direction.RIGHT: 'y',
            Direction.DOWN: 'b',
        }
        # for x_data in [x_data_left, x_data_right]:
        for x_data in [x_data_left]:
            x_data = (x_data - np.mean(x_data, 0)) / np.std(x_data, 0)
            fig = plt.figure()
            for i in range(0, 7):
                for j in range(i + 1, 7):
                    plt.subplot(6, 6, 1 + i + 6 * (j-1))
                    for direction in Direction:
                        if direction != Direction.UNDEFINED:
                            ids = np.argwhere(y_data==direction.value)
                            plt.scatter(x_data[ids,i], x_data[ids,j], c=colors[direction], label=direction.name)
            plt.figlegend(labels=colors.keys(),loc="upper right")
        return fig

    def showValidationScoreEvolution(self):
        """Plot the evolution score curve.

        Returns:
            plt.Figure: Figure created
        """
        scores = np.array(self.getValidationScoreEvolution())
        fig = plt.figure()
        if len(scores) > 0:
            plt.plot(scores[:, 0], scores[:, 1])
        plt.xlabel('Number of data in the dataset')
        plt.ylabel('Estimation success rate')
        plt.title('Cross validation evaluation')
        return fig
