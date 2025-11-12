import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load models and test data
clf_room = joblib.load('rf_room_model.pkl')
clf_time = joblib.load('rf_time_model.pkl')
X_test, y_room_test, y_time_test = joblib.load('rf_test_data.pkl')

# Predict next room and time
predicted_rooms = clf_room.predict(X_test)
predicted_times = clf_time.predict(X_test)

# Evaluate models
room_accuracy = accuracy_score(y_room_test, predicted_rooms)
time_mae = mean_absolute_error(y_time_test, predicted_times)

print(f"Random Forest Room Accuracy: {room_accuracy:.4f}")
print(f"Random Forest Time MAE (seconds): {time_mae:.2f}")

# Real-time visualization
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# --- Plot 1: Room Prediction ---
pred_line, = ax1.plot([], [], 'r--s', label='Predicted Room', linewidth=1)
actual_line, = ax1.plot([], [], 'g-o', label='Actual Room', linewidth=3)

ax1.set_xlabel('Time Step')
ax1.set_ylabel('Room Number')
ax1.set_title(f"Room Prediction (Accuracy: {room_accuracy:.4f})")
ax1.legend(loc='upper right')

# --- Plot 2: Time Prediction ---
time_pred_line, = ax2.plot([], [], 'r--', label='Predicted Time')
time_actual_line, = ax2.plot([], [], 'g-', label='Actual Time')

ax2.set_xlabel('Time Step')
ax2.set_ylabel('Time Taken (s)')
ax2.set_title(f"Time Prediction (MAE: {time_mae:.2f})")
ax2.legend(loc='upper right')

plot_open = True
def on_close(event):
    global plot_open
    plot_open = False
fig.canvas.mpl_connect('close_event', on_close)

# Simulation loop
time_steps, actual_rooms, predicted_rooms_plot = [], [], []
actual_times, predicted_times_plot = [], []

for i in range(len(y_room_test)):
    if not plot_open:
        print("Plot closed by user. Ending simulation.")
        break

    time_steps.append(i)
    actual_rooms.append(y_room_test.iloc[i])
    predicted_rooms_plot.append(predicted_rooms[i])

    actual_times.append(y_time_test.iloc[i])
    predicted_times_plot.append(predicted_times[i])

    pred_line.set_data(time_steps, predicted_rooms_plot)
    actual_line.set_data(time_steps, actual_rooms)
    time_pred_line.set_data(time_steps, predicted_times_plot)
    time_actual_line.set_data(time_steps, actual_times)

    ax1.relim(); ax1.autoscale_view()
    ax2.relim(); ax2.autoscale_view()

    plt.pause(0.5)  # Adjust delay as needed

plt.ioff()
plt.close(fig)
print("Simulation terminated.")
