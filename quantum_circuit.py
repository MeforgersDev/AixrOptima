import numpy as np

class Qubit:
    def __init__(self):
        self.state = np.array([1, 0], dtype=complex)

    def apply_gate(self, gate):
        if gate == 'x':
            X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
            self.state = np.dot(X_GATE, self.state)
        elif gate == 'hadamard':
            H_GATE = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
            self.state = np.dot(H_GATE, self.state)
        elif gate == 'phase':
            phase_angle = np.pi / 4  # Örnek faz açısı
            PHASE_GATE = np.array([[1, 0], [0, np.exp(1j * phase_angle)]])
            self.state = np.dot(PHASE_GATE, self.state)

    def measure(self):
        probabilities = np.abs(self.state) ** 2
        return np.random.choice([0, 1], p=probabilities)

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qubits = [Qubit() for _ in range(num_qubits)]

    def apply_gate(self, gate, qubit_index):
        self.qubits[qubit_index].apply_gate(gate)

    def apply_cnot(self, control_qubit, target_qubit):
        """CNOT"""
        if control_qubit < self.num_qubits and target_qubit < self.num_qubits:
            # CNOT door basic control line
            pass

    def measure_all(self):
        return [qubit.measure() for qubit in self.qubits]
