import numpy as np

def time_step_density_adjustment(time,density):
   return time*density

def time_delta_density_adjustment(delta_t,density):
   return delta_t/density

def coordinate_density_adjustment(sequence,density):
   n = len(sequence)
   print(n)
   new_sequence = np.zeros(n+(n-1)*(density-1),np.float64)
   for num in range(n-1):
      new_sequence[num+(num)*(density-1)] = sequence[num]
      for num2 in range(density-1):
        new_sequence[num+(num)*(density-1)+num2+1] = sequence[num]+(sequence[num+1]-sequence[num])/density*(num2+1)
   new_sequence[len(new_sequence)-1] = sequence[len(sequence)-1]
   return new_sequence

#   sequence = np.array(sequence)
#   n = len(sequence)
#   print(n)
#   for num in range(n-4):
#      np.insert(sequence, (num+1), (sequence[num*2]+sequence[num+1])/2)
#   return sequence

