import numpy as np


class RSESLoader:
    @staticmethod
    def load(file_path):
        """
        Static method to load data from a file.

        Parameters:
        - file_path (str): Path to the file to load.

        Returns: - tuple or None: Returns a tuple (X, y) where X is input data (numpy.ndarray) and y is target values
        (numpy.ndarray). Returns (None, None) if there was an error loading the file.
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()  # Read all lines from the file
                data = []
                for line in lines:
                    line = line.strip()  # Remove leading/trailing whitespaces
                    if line and not line.startswith('x'):  # Ignore empty lines and header line
                        values = line.split()  # Split line into values
                        data.append([float(val) for val in values])  # Convert values to float and add to data list
                data = np.array(data)  # Convert data list to NumPy array
                X = data[:, :-1]  # All columns except the last as input data
                y = data[:, -1]  # Last column as target values
                return X, y  # Return input data X and target values y
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return None, None  # Return None if file is not found
        except Exception as e:
            print(f"Error loading file '{file_path}': {str(e)}")
            return None, None  # Return None if there is any other exception
