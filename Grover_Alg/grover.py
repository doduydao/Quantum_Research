from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.circuit.library import C4XGate, C3XGate
from qiskit.visualization import plot_distribution, plot_histogram
import matplotlib.pyplot as plt

def initialization():
    qreg_q = QuantumRegister(5, 'q') # Define 5_qubits
    creg_c = ClassicalRegister(4, 'c') # Define 4 registers
    circuit = QuantumCircuit(qreg_q, creg_c) # Define quantum circuit
    circuit.x(4) # Apply X gate to qubit 4
    circuit.barrier(range(5)) # Barriers
    circuit.h(range(5)) # Apply Hardarmad gate to all qubit
    circuit.barrier(range(5))
    return circuit


def oracle():
    circuit = QuantumCircuit(5)
    circuit.x(0)
    circuit.x(2)
    circuit.append(C4XGate(), [0,1,2,3,4]) # CCCCCX gate
    circuit.x(0)
    circuit.x(2)
    circuit.barrier(range(5))
    return circuit

def amplitude():
    circuit = QuantumCircuit(5)
    circuit.h(range(4))
    circuit.x(range(4))
    circuit.barrier(range(3))

    # CCCCZ Gate
    circuit.h(3)
    circuit.append(C3XGate(), [0,1,2,3])
    circuit.h(3)

    circuit.barrier(range(3))
    circuit.x(range(4))
    circuit.h(range(4))
    circuit.barrier(range(5))
    return circuit

def Grover(no_iterations, no_sampling):
    circuit = initialization()
    circuit_oracle = oracle()
    circuit_amplitude = amplitude()

    for _ in range(no_iterations):
        circuit &= circuit_oracle & circuit_amplitude

    circuit.measure(range(4), range(4)) # Add measuring
    # circuit.draw("mpl") # Draw the circuit
    # plt.show()
    simulateQasm(circuit, no_sampling)

# simulate in qasm_simulator
def simulateQasm(circuit, no_sampling):
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend, shots=no_sampling).result()
    counts = result.get_counts()
    plot_histogram(counts, color='RoyalBlue', title="Qasm Histogram")
    plot_distribution(counts, color='RoyalBlue', title="Qasm Distribution")
    plt.show()

if __name__ == '__main__':
    Grover(no_interations=3, no_sampling=1000)


