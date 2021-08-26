# Quaternion Conjugate
def quaternConj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype = np.float64)

# Quaternion Product
def quaternProd(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

# Quaternions to Euler Angles
def quatern2eul(q):
        N = q.shape[0]
        roll_x = np.zeros(N)
        pitch_y = np.zeros(N)
        yaw_z = np.zeros(N)
        
        for k in range(0, N):
            w, x, y, z = q[k,:]
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x[k] = math.atan2(t0, t1)

            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y[k] = math.asin(t2)

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z[k] = math.atan2(t3, t4)
         
        return roll_x, pitch_y, yaw_z # [rad]

# Euler Angles to Quaternions
def eul2quatern(roll_x, pitch_y, yaw_z):
    N = roll_x.shape[0]
    q = np.zeros((N,4))
    
    for k in range(0,N):
        cy, sy = np.cos(yaw_z[k] * .5), np.sin(yaw_z[k] * .5)
        cp, sp = np.cos(pitch_y[k] * .5), np.sin(pitch_y[k] * .5)
        cr, sr = np.cos(roll_x[k] * .5), np.sin(roll_x[k] * .5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        q[k,:] = w, x, y, z
    
    return q

# --- Simple Complementary Filter 
def compl_filt(acc_data, gyr_data, fs, beta):
    N = acc_data.shape[0]
    roll_x, pitch_y, yaw_z, angle_x, angle_y, angle_z = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    dk = 1/fs
    
    for k in range(1,N):

        alpha_x = np.arctan2(acc_data[k,1], acc_data[k,2])
        alpha_y = np.arctan2(-acc_data[k,0],  np.sqrt( np.square(acc_data[k,1]) + np.square(acc_data[k,2]) ))

        angle_x[k] = 0.5 * ( gyr_data[k-1,0] + gyr_data[k,0] ) + angle_x[k-1]
        angle_y[k] = 0.5 * ( gyr_data[k-1,1] + gyr_data[k,1] ) + angle_y[k-1]
        angle_z[k] = 0.5 * ( gyr_data[k-1,2] + gyr_data[k,2] ) + angle_z[k-1]

        roll_x[k] = (1 - beta) * ( angle_x[k-1] + gyr_data[k,0] * dk ) + beta * alpha_x
        pitch_y[k] = (1 - beta) * ( angle_y[k-1] + gyr_data[k,1] * dk ) + beta * alpha_y
        yaw_z[k] = angle_z[k] 
        
    return roll_x*np.pi/180, pitch_y*np.pi/180, yaw_z*np.pi/180 # Angles in [rad]

def align_to_WRF(q, acc):
    N = acc.shape[0]
    a_q, q_star, q_glob = np.zeros((N,4)), np.zeros((N,4)), np.zeros((N,4))
    a_q[:,1:] = acc

    for t in range(0,N):
        q_star[t] = quaternConj(q[t])
        q_temp = quaternProd(q[t], a_q[t])
        q_glob[t] = quaternProd(q_temp, q_star[t])

    return q_glob[:,2] - np.mean(q_glob[0:100,2])