"""
adapted from
https://github.com/torchmd/mdgrad/tree/master/nff/md
"""
import numpy as np
from ase.md.md import MolecularDynamics
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
import math
from copy import copy
from ase import units

class NoseHoover(MolecularDynamics):
    def __init__(self,
                 atoms,
                 timestep,
                 constraints,
                 increm,
                 temperature,
                 ttime,
                 Mne,
                 eta_length,
                 targetmu,
                 f0=None,
                 trajectory=None,
                 logfile=None,
                 loginterval=1,
                 **kwargs):

        super().__init__(
                         atoms,
                         timestep,
                         trajectory,
                         logfile,
                         loginterval)

        # Initialize simulation parameters

        self.dt = timestep
        self.delT = self.dt/units.fs
        self.m = np.array(self.atoms.get_masses())
        self.constraints = constraints
        self.shaketol = 0.00001
        self.shakemaxiter = 1000
        self.increm = increm
        self.cvgd = []

        self.Natom = atoms.get_number_of_atoms()
        self.T = temperature
        self.targeEkin = 0.5 * (3.0 * self.Natom) * self.T
        self.ttime = ttime  # * units.fs
        self.Q = 3.0 * self.Natom * self.T * (self.ttime * self.dt)**2
        self.zeta = 0.0
        self.lagrange = np.zeros(len(constraints))

        self.istep = 0

        # fluctuation Fermi level

        self.Vne = np.zeros(3)
        self.Mne = Mne 
        self.M_eta = 8.6173E-5 * self.T / units.kB * 81.27  
        self.nc = 1
        self.M = eta_length
        self.Vlogs = np.zeros(self.M)
        self.Xlogs = np.zeros(self.M)
        self.Glogs = np.zeros(self.M)
        self.Qmass = np.ones(self.M) * self.M_eta
        self.ne = atoms.info['electron']
        self.targetmu = targetmu
        self.mu = 0

    def constrained_md(self):
        x_old = self.atoms.get_positions()
        v_old = np.array(self.atoms.get_velocities())
        m = self.m
        dt = self.dt
        constraints = self.constraints
        shaketol = self.shaketol
        shakemaxiter = self.shakemaxiter

        lagrange = np.zeros(len(constraints))
        itr = 0
        noa = len(m)
        converged = 1
        damp_coeff = 1
        Z = 0

        x_new = x_old + v_old * dt
        v_new = copy(v_old)

        while((not all(abs(self.sigma(c, x_new)) < shaketol for c in constraints) and itr < shakemaxiter) or itr == 0):

            for k in range(0, len(constraints)):

                denominator = 0
                for i in range(1, len(constraints[k])-1):
                    denominator += np.dot(self.sigma_del(constraints[k], i, x_old), self.sigma_del(constraints[k], i, x_new)) / m[constraints[k][i]-1]
                if denominator != 0: lagrange[k] = self.sigma(constraints[k], x_new) / (denominator * dt**2)
                else: lagrange[k] = 0
                self.lagrange[k] += self.sigma(constraints[k], x_new) / (denominator * dt**2)

            for i in range(0, noa):
                for k in range(0, len(constraints)):

                    if (i+1) in constraints[k][1:-1]:
                        speed = self.distance(v_new[i], np.zeros(3))
                        v_new[i] += - damp_coeff * (lagrange[k] * dt / m[i]) * self.sigma_del(constraints[k], constraints[k][1:-1].index(i+1) + 1, x_new)
            x_new = x_old + v_new * dt

            itr += 1
            if itr == 1000: converged = 0

        for i in range(0,noa):
            if (i+1) in constraints[0][1:-1]:
                Z += np.dot(self.sigma_del(constraints[0], constraints[0][1:-1].index(i+1) + 1, x_new),self.sigma_del(constraints[0], constraints[0][1:-1].index(i+1) + 1, x_new))/m[i]

        print("Gradient = " + str(-self.lagrange[-1]))
        return v_new, converged

    def Ekin(self, v, m):
        '''
        The force of harmonic potential
        '''

        #return 103.65 * 0.5 * sum(sum(np.transpose(v * v)) * m)
        return  0.5 * m * v**2
    '''
    def log(self):
        i = self.istep
        dt = self.dt
        x = self.atoms.get_positions()
        v = self.atoms.get_velocities()
        f = self.atoms.get_forces()
        m = self.atoms.get_masses().reshape(-1, 1)

        nh_out = open("MD_out", "a")
        nh_out.write("Step " + str(i) + "  /  Time (fs) " + str(dt * i) + "\n")
        nh_out.write("Positions (Cartesian)              Velocities (A/fs)                Forces (eV/A)\n")
        for i in range(0, len(x)):
            nh_out.write('%.6f' % x[i][0] + "   " + '%.6f' % x[i][1] + "   " + '%.6f' % x[i][2] + "   ")
            nh_out.write('%.6f' % v[i][0] + "   " + '%.6f' % v[i][1] + "   " + '%.6f' % v[i][2] + "   ")
            nh_out.write('%.6f' % f[i][0] + "   " + '%.6f' % f[i][1] + "   " + '%.6f' % f[i][2] + "\n")

        nh_out.write("Kinetic Energy (eV): " + str(self.Ekin(v, m)) + "\n")
        kb = 0.0000861733034
        nh_out.write("Temperature (K): " + str(2 * self.Ekin(v, m) / ((3 * len(x) - 3) * kb)) + "\n\n")
        nh_out.close()

        if (i%10 == 0):
            print("Completed MD step = ", i)
    '''

    def distance(self, v1, v2):
        dist = 0
        for i in range(0, len(v1)):
            dist += (v1[i] - v2[i])**2
        return math.sqrt(dist)

    def sigma(self, constraint, positions):
        if constraint[0] == 0:
            pos1 = positions[constraint[1] - 1]
            pos2 = positions[constraint[2] - 1]
            return self.distance(pos1, pos2) - constraint[-1]
        elif constraint[0] == 1:
            pos_i = positions[constraint[1] - 1]
            pos_j = positions[constraint[2] - 1]
            pos_k = positions[constraint[3] - 1]
            return self.distance(pos_i, pos_j) - self.distance(pos_j, pos_k) - constraint[-1]

    def sigma_del(self, constraint, index, positions):
        if constraint[0] == 0:
            pos1 = np.array(positions[constraint[1] - 1])
            pos2 = np.array(positions[constraint[2] - 1])
            if index == 1:  # derivative wrt pos1
                return (pos1 - pos2) / self.distance(pos1, pos2)
            else:  # derivative wrt pos2
                return (pos2 - pos1) / self.distance(pos1, pos2)
        elif constraint[0] == 1:
            pos_i = np.array(positions[constraint[1] - 1])
            pos_j = np.array(positions[constraint[2] - 1])
            pos_k = np.array(positions[constraint[3] - 1])
            if index == 1:  # derivative wrt pos_i
                return (pos_i - pos_j) / (self.distance(pos_i, pos_j))
            elif index == 2:  # derivative wrt pos_j
                return -(pos_i - pos_j) / (self.distance(pos_i, pos_j)) - (pos_j - pos_k) / (self.distance(pos_j, pos_k))
            elif index == 3:  # derivative wrt pos_k
                return (pos_j - pos_k) / (self.distance(pos_j, pos_k))

    def nhc_step(self, v, Glogs, Vlogs, Xlogs):

        T = self.T/units.kB/5
        Qmass = self.Qmass
        k_b = 8.6173E-5    
        delT = self.delT
        scale = 1.0
        M = Glogs.size
        K = self.Ekin(v, self.Mne)
        K2 = 2*K
        Glogs[0] = (K2 - k_b * T) / Qmass[0]
        tmp = 1 / (2 - 2**(1./3))
        wdti = np.array([tmp, 1 - 2*tmp, tmp]) * delT / self.nc
    
        for inc in range(self.nc):
            for iys in range(3):
                wdt = wdti[iys]
                # update the thermostat velocities
                Vlogs[-1] += 0.25 * Glogs[-1] * wdt

                for kk in range(M-1):
                    AA = np.exp(-0.125 * wdt * Vlogs[M-1-kk])
                    Vlogs[M-2-kk] = Vlogs[M-2-kk] * AA * AA \
                                  + 0.25 * wdt * Glogs[M-2-kk] * AA

                # update the particle velocities
                AA = np.exp(-0.5 * wdt * Vlogs[0])
                scale *= AA
                # update the forces
                #print("K2 ", K2)
                #print("Glogs: ", scale * scale * K2 - k_b * T)
                Glogs[0] = (scale * scale * K2 - k_b * T) / Qmass[0]
                # update the thermostat positions
                Xlogs += 0.5 * Vlogs * wdt
                # update the thermostat velocities
                for kk in range(M-1):
                    AA = np.exp(-0.125 * wdt * Vlogs[kk+1])
                    Vlogs[kk] = Vlogs[kk] * AA * AA \
                              + 0.25 * wdt * Glogs[kk] * AA
                    Glogs[kk+1] = (Qmass[kk] * Vlogs[kk]**2 - k_b * T) / Qmass[kk+1]
                Vlogs[-1] += 0.25 * Glogs[-1] * wdt
        return v * scale

    def step(self):

        # get current acceleration and velocity:
        accel = self.atoms.get_forces() / self.atoms.get_masses().reshape(-1, 1)
        vel = self.atoms.get_velocities()

        # record current velocities
        KE_0 = self.atoms.get_kinetic_energy()

        # make half a step in velocity
        vel_half = vel + 0.5 * self.dt * (accel - self.zeta * vel)
        self.atoms.set_velocities(vel_half)

        # constrained
        converged = 1
        self.lagrange = np.zeros(len(self.constraints))
        if self.constraints and len(self.constraints[0]) > 0: 
            vel_half, converged = self.constrained_md()
            #self.constraints[0][3] = self.constraints[0][3] + self.increm
            #print("distance: " + str(self.constraints[0][3]))
            for c in self.constraints:
                c[-1] += self.increm
            print("distance: " + str(self.constraints[0][-1]))

        self.cvgd.append(converged)
        self.atoms.set_velocities(vel_half)

        # make full step in position
        x = self.atoms.get_positions() + vel_half * self.dt
        self.atoms.set_positions(x)

        self.Vne[0] = self.nhc_step(self.Vne[0], self.Glogs, self.Vlogs, self.Xlogs)
        self.Vne[1] = self.Vne[0] + (self.targetmu - self.atoms.get_calculator().get_mu()) * self.delT / 2 / self.Mne
        self.ne = self.ne + self.Vne[1] * self.delT
        self.atoms.info['electron'] = np.array(self.ne)

        # make a full step in accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)

        print('ne:', str(self.atoms.info['electron']))

        self.Vne[2] = self.Vne[1] + (self.targetmu - self.atoms.get_calculator().get_mu()) * self.delT / 2 / self.Mne
        self.Vne[2] = self.nhc_step(self.Vne[2], self.Glogs, self.Vlogs, self.Xlogs)
        self.Vne[0] = self.Vne[2]

        # make a half step in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1/self.Q) * (KE_0 - self.targeEkin)

        # make another halfstep in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * \
            (1/self.Q) * (self.atoms.get_kinetic_energy() - self.targeEkin)

        # make another half step in velocity
        vel = (self.atoms.get_velocities() + 0.5 * self.dt * accel) / \
            (1 + 0.5 * self.dt * self.zeta)
        self.atoms.set_velocities(vel)

        return f

