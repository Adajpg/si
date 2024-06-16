import numpy as np

class RSESLoader:
    @staticmethod
    def load(file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                data = []
                for line in lines[1:]:  # Pomijamy pierwszy wiersz z nagłówkiem
                    line = line.strip()
                    if line:  # Ignoruj puste wiersze
                        values = line.split()  # Podziel wiersz na wartości oddzielone spacją lub tabulatorem
                        data.append([float(val) for val in values])
                data = np.array(data)
                X = data[:, :-1]  # Wszystkie kolumny oprócz ostatniej jako dane wejściowe
                y = data[:, -1]  # Ostatnia kolumna jako wartości docelowe
                return X, y
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return None, None
        except Exception as e:
            print(f"Error loading file '{file_path}': {str(e)}")
            return None, None
