from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt

qreg_q = QuantumRegister(3, 'q')
creg_c = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q)
circuit.h(qreg_q)

theta_x = 1
theta_z = 0
delta = 0.1
h_z = 3/2
no_iter = int(theta_x / delta)

for iter in range(no_iter):
	print(iter + 1, " ", str(theta_x))

	circuit.rx(theta_x, qreg_q)

	# Rotation of Z_1.Z_2
	# (-1/2).Z_1.Z_2
	t = -0.5/h_z
	circuit.cx(qreg_q[0], qreg_q[1])
	circuit.rz(2 * t * theta_z, qreg_q[1])
	circuit.cx(qreg_q[0], qreg_q[1])

	# Rotation of Z_1.Z_3
	# (-1/2).Z_1.Z_3
	t = -0.5 / h_z
	circuit.cx(qreg_q[0], qreg_q[2])
	circuit.rz(2 * t * theta_z, qreg_q[2])
	circuit.cx(qreg_q[0], qreg_q[2])

	# Rotation of Z_2.Z_3
	# (-1/2).Z_2.Z_3
	t = -0.5 / h_z
	circuit.cx(qreg_q[1], qreg_q[2])
	circuit.rz(2 * t * theta_z, qreg_q[2])
	circuit.cx(qreg_q[1], qreg_q[2])

	# update angle
	theta_x -= delta
	theta_z += delta

circuit.measure(qreg_q, creg_c)
circuit.draw("mpl")
plt.show()
backend = Aer.get_backend('qasm_simulator')
NUM_SHOTS = 1000
job = execute(circuit, backend, shots=NUM_SHOTS)  # NUM_SHOTS
result = job.result()

print("time_taken = ", result.time_taken)
counts = result.get_counts(circuit)
counts = {k[::-1]:v for k, v in counts.items()} # Reverse qubit order
print("count = ", counts)
fig_counted = plot_histogram(counts)
fig_proba = plot_distribution(counts,title="Qasm Distribution")
fig_counted.savefig('counted_result.png')
fig_proba.savefig('proba_result.png')