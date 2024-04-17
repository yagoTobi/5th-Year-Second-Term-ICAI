import numpy as np

state = {
    0: "Recibir invitación amigo",
    1: "Visitar página web principal",
    2: "Darse de alta con correo",
    3: "Descargarse el navegador",
    4: "Finalizar el proceso onboarding",
    5: "Usar navegador 6/7",
}

A = np.array(
    [
        # ? - See diagram as reference
        [0.2, 0.0, 0.8, 0.0, 0.0, 0.0],  # * Recibir invitación de un amigo
        [0.0, 0.2, 0.8, 0.0, 0.0, 0.0],  # * Visitar página web principal esta semana
        [0.2, 0.2, 0.0, 0.6, 0.0, 0.0],  # * Darse de alta con el correo
        [0.1, 0.1, 0.1, 0.0, 0.7, 0.0],  # * Descargar el navegador
        [0.2, 0.2, 0.0, 0.0, 0.0, 0.6],  # * Finalizar el proceso onboarding
        [0.1, 0.1, 0.0, 0.0, 0.5, 0.3],  # * Usar el navegador 6/7 días
    ]
)

n = 10
start_state = 0
print(state[start_state], "-->", end=" ")
prev_state = start_state

while n - 1:
    curr_state = np.random.choice([0, 1, 2, 3, 4, 5], p=A[prev_state])
