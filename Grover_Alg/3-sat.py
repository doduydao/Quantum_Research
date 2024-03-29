from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.circuit.library import C3XGate
from qiskit.visualization import plot_distribution, plot_histogram
import matplotlib.pyplot as plt

def initialization():
    qreg_q = QuantumRegister(7, 'q')
    creg_c = ClassicalRegister(3, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    circuit.x(-1)
    circuit.barrier(range(7))
    circuit.h(range(3))
    circuit.h(-1)
    circuit.barrier(range(7))
    return circuit


def oracle():
    circuit = QuantumCircuit(7)
    # constraint 1: x_1 ^ x_2 ^ ~x_3
    circuit.x(2)
    circuit.cx(range(0, 3),3)
    circuit.append(C3XGate(),[0,1,2,3])
    circuit.x(2)
    circuit.barrier(range(5))
    # constraint 2: ~x_1 ^ ~x_2 ^ ~x_3
    circuit.x(range(0, 3))
    circuit.cx(range(0, 3), 4)
    circuit.append(C3XGate(), [0, 1, 2, 4])
    circuit.x(range(0, 3))
    circuit.barrier(range(5))
    # constraint 3: ~x_1 ^ x_2 ^ x_3
    circuit.x(0)
    circuit.cx(range(0, 3), 5)
    circuit.append(C3XGate(), [0, 1, 2, 5])
    circuit.x(0)
    circuit.barrier(range(5))

    circuit_oracle_reverse = circuit.reverse_ops() # Reverse of c_1, c_2 and c_3
    # c_1 & c_2 & c3
    circuit.append(C3XGate(), [3, 4, 5, 6])
    circuit.barrier(range(5))
    circuit &= circuit_oracle_reverse
    return circuit

def amplitude():
    circuit = QuantumCircuit(7)
    circuit.h(range(3))
    circuit.x(range(3))
    circuit.barrier(range(3))
    circuit.h(2)
    circuit.ccx(0,1,2)
    circuit.h(2)
    circuit.barrier(range(3))
    circuit.x(range(3))
    circuit.h(range(3))
    circuit.barrier(range(7))
    return circuit

def Grover(no_interations, no_sampling):
    circuit = initialization()
    circuit_oracle = oracle()
    circuit_ampliture = amplitude()

    for _ in range(no_interations):
        circuit &= circuit_oracle & circuit_ampliture
    circuit.measure(range(3), range(3))
    # circuit.draw("mpl")
    # plt.show()
    simulateQasm(circuit, no_sampling)

# simulate in qasm_simulator
def simulateQasm(circuit, no_sampling):
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend, shots=no_sampling).result()
    counts = result.get_counts()
    plot_histogram(counts, color='midnightblue', title="Qasm Histogram")
    plot_distribution(counts, color='midnightblue', title="Qasm Distribution")
    plt.show()

if __name__ == '__main__':
    Grover(no_interations=2, no_sampling=1024)


