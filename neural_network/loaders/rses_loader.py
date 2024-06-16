import numpy as np


class RSESLoader:
    @staticmethod
    def load(file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                data = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('x'):  # Ignore empty lines and header line
                        values = line.split()
                        data.append([float(val) for val in values])
                data = np.array(data)
                X = data[:, :-1]  # All columns except the last as input data
                y = data[:, -1]  # Last column as target values
                return X, y
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return None, None
        except Exception as e:
            print(f"Error loading file '{file_path}': {str(e)}")
            return None, None