"""
Classes to deal with ouputs from tracking algorithm individually and in groups
"""


class TrackedPoint(object):
    """
    Individual marker class
    with x, y, coordinates and the likelihoods
    """

    # instance attribute
    def __init__(self, name, x, y, likelihood):
        # attributes
        self.name = name
        self.x = x
        self.y = y
        self.likelihood = likelihood

    @property
    def get_name(self):
        return self.name

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_likelihood(self):
        return self.likelihood


class TrackedFingers(TrackedPoint):
    """
    Group class for fingers
    """

    def __init__(self, df, parts_group_label):

        self.parts = []

        def read_vals_key(df, body_part):
            x = df[df.keys()[0][0], body_part, "x"].values
            y = df[df.keys()[0][0], body_part, "y"].values
            likelihood = df[df.keys()[0][0], body_part, "likelihood"].values
            return x, y, likelihood

        for bodypart in parts_group_label:
            self.parts.append(TrackedPoint(bodypart, *read_vals_key(df, bodypart)))


class TrackedGroup(object):
    """
    Group class for multiple markers
    with x, y, coordinates and the likelihoods
    """

    # instance attribute
    def __init__(self, name, x, y, likelihood):
        # attributes
        # list of names
        self.name = name
        # array of x coordinates
        self.x = x
        # array of y coordinates
        self.y = y
        # array of likelihoods
        self.likelihood = likelihood

    @property
    def get_name(self):
        return self.name

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_likelihood(self):
        return self.likelihood

    def get_dims(self):
        return self.x.shape

    def get_number_features(self):
        return self.x.shape[0]

    def get_number_timeframes(self):
        return self.x.shape[1]
