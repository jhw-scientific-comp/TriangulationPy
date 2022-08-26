from tkinter import *
from tkinter import ttk
from tkinter.ttk import *
import numpy as np
import math
import itertools



class Domain:
	def __init__(self, v_boundary_var):
		self.v_boundary = v_boundary_var
		self.e_boundary = []
		self.e_boundary_indices = []

		for i in range(0, len(self.v_boundary)-1):
			self.e_boundary.append((self.v_boundary[i], self.v_boundary[i + 1]))
			self.e_boundary_indices.append((i, i+1))
		self.e_boundary.append((self.v_boundary[len(self.v_boundary)-1], self.v_boundary[0]))
		self.e_boundary_indices.append((len(self.v_boundary)-1, 0))

	def vrange(self, start, end, reverse_order):
		i = start
		j = 0
		stop = 0		

		if reverse_order == False:
			if start <= end:
				stop = end - start
			else:
				stop = len(self.v_boundary) - start + end

			while j <= stop:
				yield i
				j += 1
				if i == len(self.v_boundary) - 1:
					i = 0
				else:
					i += 1
		else:
			if start < end:
				stop = len(self.v_boundary) + start - end
			else:
				stop = start - end
			
			while j <= stop:
				yield i
				j += 1
				if i == 0:
					i = len(self.v_boundary) - 1
				else:
					i -= 1

	def getVBoundary(self):
		return self.v_boundary

	def getEBoundary(self):
		return self.e_boundary

	def getEBoundaryIndices(self):
		return self.e_boundary_indices



def det2D(m):
	return (m[0][0] * m[1][1] - m[1][0] * m[0][1])

# def det(m):
	# det_val = 0.0

	# # if m.shape[0] == 3:
		# # for j in range(0, 3):
			# # det_val = pow(-1.0, j+1) * m[0][j] * det2D(np.array())
	# # else:

	# return det_val

	

def intersect(e1, e2):	
	A = np.array([[e2[0][0] - e1[0][0], e2[1][0] - e1[0][0]], [e2[0][1] - e1[0][1], e2[1][1] - e1[0][1]]])
	B = np.array([[e2[0][0] - e1[1][0], e2[1][0] - e1[1][0]], [e2[0][1] - e1[1][1], e2[1][1] - e1[1][1]]])
	C = np.array([[e1[0][0] - e2[0][0], e1[1][0] - e2[0][0]], [e1[0][1] - e2[0][1], e1[1][1] - e2[0][1]]])
	D = np.array([[e1[0][0] - e2[1][0], e1[1][0] - e2[1][0]], [e1[0][1] - e2[1][1], e1[1][1] - e2[1][1]]])

	
	if (abs(det2D(A)) < np.finfo(float).eps) and (abs(det2D(B)) < np.finfo(float).eps) and (abs(det2D(C)) < np.finfo(float).eps) and (abs(det2D(D)) < np.finfo(float).eps) and\
	(((e1[0][0] >= min(e2[0][0], e2[1][0])) and (e1[0][0] <= max(e2[0][0], e2[1][0])) and (e1[0][1] >= min(e2[0][1], e2[1][1])) and (e1[0][1] <= max(e2[0][1], e2[1][1]))) or\
	((e1[1][0] >= min(e2[0][0], e2[1][0])) and (e1[1][0] <= max(e2[0][0], e2[1][0])) and (e1[1][1] >= min(e2[0][1], e2[1][1])) and (e1[1][1] <= max(e2[0][1], e2[1][1])))):
		#return True


		if (((abs(min(e1[0][0], e1[1][0]) - max(e2[0][0], e2[1][0])) < np.finfo(float).eps) and (abs(min(e1[0][1], e1[1][1]) - max(e2[0][1], e2[1][1])) < np.finfo(float).eps)\
		and (((max(e1[0][0], e1[1][0]) >= max(e2[0][0], e2[1][0])) and (max(e1[0][1], e1[1][1]) > max(e2[0][1], e2[1][1]))) or (((max(e1[0][0], e1[1][0]) > max(e2[0][0], e2[1][0])) and (max(e1[0][1], e1[1][1]) >= max(e2[0][1], e2[1][1]))))))\
		or ((abs(max(e1[0][0], e1[1][0]) - min(e2[0][0], e2[1][0])) < np.finfo(float).eps) and (abs(min(e1[0][1], e1[1][1]) - max(e2[0][1], e2[1][1])) < np.finfo(float).eps)\
		and (((min(e1[0][0], e1[1][0]) <= min(e2[0][0], e2[1][0])) and (max(e1[0][1], e1[1][1]) > max(e2[0][1], e2[1][1]))) or (((min(e1[0][0], e1[1][0]) < min(e2[0][0], e2[1][0])) and (max(e1[0][1], e1[1][1]) >= max(e2[0][1], e2[1][1]))))))\
		or ((abs(max(e1[0][0], e1[1][0]) - min(e2[0][0], e2[1][0])) < np.finfo(float).eps) and (abs(max(e1[0][1], e1[1][1]) - min(e2[0][1], e2[1][1])) < np.finfo(float).eps)\
		and (((min(e1[0][0], e1[1][0]) <= min(e2[0][0], e2[1][0])) and (min(e1[0][1], e1[1][1]) < max(e2[0][1], e2[1][1]))) or (((min(e1[0][0], e1[1][0]) < min(e2[0][0], e2[1][0])) and (min(e1[0][1], e1[1][1]) <= min(e2[0][1], e2[1][1]))))))\
		or ((abs(min(e1[0][0], e1[1][0]) - max(e2[0][0], e2[1][0])) < np.finfo(float).eps) and (abs(max(e1[0][1], e1[1][1]) - min(e2[0][1], e2[1][1])) < np.finfo(float).eps)\
		and (((max(e1[0][0], e1[1][0]) >= max(e2[0][0], e2[1][0])) and (min(e1[0][1], e1[1][1]) < min(e2[0][1], e2[1][1]))) or (((max(e1[0][0], e1[1][0]) > max(e2[0][0], e2[1][0])) and (min(e1[0][1], e1[1][1]) <= min(e2[0][1], e2[1][1]))))))):
			return 1 #collinear intersection without overlap
		else:
			return 3 #collinear intersection with overlap
	elif (det2D(A) <= 0.0) and (det2D(B) > 0.0) and (det2D(C) <= 0.0) and (det2D(D) > 0.0):
		#return True
		return 2
	elif (det2D(A) <= 0.0) and (det2D(B) > 0.0) and (det2D(C) < 0.0) and (det2D(D) >= 0.0):
		#return True
		return 2
	elif (det2D(A) <= 0.0) and (det2D(B) > 0.0) and (det2D(C) >= 0.0) and (det2D(D) < 0.0):
		#return True
		return 2
	elif (det2D(A) <= 0.0) and (det2D(B) > 0.0) and (det2D(C) > 0.0) and (det2D(D) <= 0.0):
		#return True
		return 2
	elif (det2D(A) >= 0.0) and (det2D(B) < 0.0) and (det2D(C) <= 0.0) and (det2D(D) > 0.0):
		#return True
		return 2
	elif (det2D(A) >= 0.0) and (det2D(B) < 0.0) and (det2D(C) < 0.0) and (det2D(D) >= 0.0):
		#return True
		return 2
	elif (det2D(A) >= 0.0) and (det2D(B) < 0.0) and (det2D(C) >= 0.0) and (det2D(D) < 0.0):
		#return True
		return 2
	elif (det2D(A) >= 0.0) and (det2D(B) < 0.0) and (det2D(C) > 0.0) and (det2D(D) <= 0.0):
		#return True
		return 2
	elif (det2D(A) < 0.0) and (det2D(B) >= 0.0) and (det2D(C) <= 0.0) and (det2D(D) > 0.0):
		#return True
		return 2
	elif (det2D(A) < 0.0) and (det2D(B) >= 0.0) and (det2D(C) < 0.0) and (det2D(D) >= 0.0):
		#return True
		return 2
	elif (det2D(A) < 0.0) and (det2D(B) >= 0.0) and (det2D(C) >= 0.0) and (det2D(D) < 0.0):
		#return True
		return 2
	elif (det2D(A) < 0.0) and (det2D(B) >= 0.0) and (det2D(C) > 0.0) and (det2D(D) <= 0.0):
		#return True
		return 2
	elif (det2D(A) > 0.0) and (det2D(B) <= 0.0) and (det2D(C) <= 0.0) and (det2D(D) > 0.0):
		#return True
		return 2
	elif (det2D(A) > 0.0) and (det2D(B) <= 0.0) and (det2D(C) < 0.0) and (det2D(D) >= 0.0):
		#return True
		return 2
	elif (det2D(A) > 0.0) and (det2D(B) <= 0.0) and (det2D(C) >= 0.0) and (det2D(D) < 0.0):
		#return True
		return 2
	elif (det2D(A) > 0.0) and (det2D(B) <= 0.0) and (det2D(C) > 0.0) and (det2D(D) <= 0.0):
		#return True
		return 2
	else:
		#return False
		return 0 #no intersection
	



def checkIntersection(e, E):
	intersection_exists = False
	i = 0

	intersection_type = 0


	while (intersection_exists == False) and (i <= len(E) - 1):

		intersection_type = intersect(e, E[i])

		if (((abs(e[0][0] - E[i][0][0]) < np.finfo(float).eps) and (abs(e[0][1] - E[i][0][1]) < np.finfo(float).eps) and ((abs(e[1][0] - E[i][1][0]) >= np.finfo(float).eps) or (abs(e[1][1] - E[i][1][1]) >= np.finfo(float).eps)))\
		or ((abs(e[1][0] - E[i][0][0]) < np.finfo(float).eps) and (abs(e[1][1] - E[i][0][1]) < np.finfo(float).eps) and ((abs(e[0][0] - E[i][1][0]) >= np.finfo(float).eps) or (abs(e[0][1] - E[i][1][1]) >= np.finfo(float).eps)))\
		or ((abs(e[0][0] - E[i][1][0]) < np.finfo(float).eps) and (abs(e[0][1] - E[i][1][1]) < np.finfo(float).eps) and ((abs(e[1][0] - E[i][0][0]) >= np.finfo(float).eps) or (abs(e[1][1] - E[i][0][1]) >= np.finfo(float).eps)))\
		or ((abs(e[1][0] - E[i][1][0]) < np.finfo(float).eps) and (abs(e[1][1] - E[i][1][1]) < np.finfo(float).eps) and ((abs(e[0][0] - E[i][0][0]) >= np.finfo(float).eps) or (abs(e[0][1] - E[i][0][1]) >= np.finfo(float).eps)))):
			
			if intersection_type == 3:
				intersection_exists = True
			else:
				intersection_exists = False
			#print("if: e = ", e, ", E[i]: ", E[i])			
		else:
			#print("else: e = ", e, ", E[i]: ", E[i])			
			
			if intersection_type == 2:
				intersection_exists = True
			else:
				intersection_exists = False
			
		#print("    ", e, ", ", E[i], ": ", intersection_exists)
		i = i + 1	

	return intersection_exists




def transformCoord(window, V_domain_var, p): 	
	x_min = min(V_domain_var, key = lambda x: x[0])[0]
	x_max = max(V_domain_var, key = lambda x: x[0])[0]
	y_min = min(V_domain_var, key = lambda x: x[1])[1]
	y_max = max(V_domain_var, key = lambda x: x[1])[1]
	
	window.update_idletasks()	
		
	delta_x = x_max - x_min
	delta_y = y_max - y_min

	if delta_x >= delta_y:
		return (round(window.winfo_width() * (0.05 - 0.9 * (x_min - p[0])/delta_x)), round( (window.winfo_height()/2.0 - window.winfo_width() * 0.5 * delta_y * 0.9/delta_x + y_max * window.winfo_width() * 0.9/delta_x) - window.winfo_width() * (p[1] * 0.9/delta_x)))
	else:
		return (round(window.winfo_width()/2.0 - 0.5 * delta_x * window.winfo_height() * 0.9/delta_y - (x_min - p[0]) * window.winfo_height() * 0.9/delta_y), round(window.winfo_height() * (0.95 + (y_min - p[1]) * 0.9/delta_y)))
	


#TODO: Exception, falls Kanten keinen gemeinsamen Punkt haben
def getAngle(e1, e2):
	angle = 0.0
	i_1 = 0
	i_2 = 0
	v = (0,0)
		
	if (abs(e1[0][0] - e2[0][0]) < np.finfo(float).eps) and (abs(e1[0][1] - e2[0][1]) < np.finfo(float).eps):
		v = e1[0]
		i_1 = 1
		i_2 = 1
	elif (abs(e1[0][0] - e2[1][0]) < np.finfo(float).eps) and (abs(e1[0][1] - e2[1][1]) < np.finfo(float).eps):
		v = e1[0]
		i_1 = 1
		i_2 = 0
	elif (abs(e1[1][0] - e2[0][0]) < np.finfo(float).eps) and (abs(e1[1][1] - e2[0][1]) < np.finfo(float).eps):
		v = e1[1]
		i_1 = 0
		i_2 = 1
	else:
		v = e1[1]
		i_1 = 0
		i_2 = 0

	#print(v)

	
	phi_1 = 180.0/math.pi * np.arccos(abs(e1[i_1][1] - v[1])/calculateEuclideanNorm2D(tuple(np.subtract(e1[i_1], v))))
	phi_2 = 180.0/math.pi * np.arccos(abs(e2[i_2][1] - v[1])/calculateEuclideanNorm2D(tuple(np.subtract(e2[i_2], v))))

	#print(phi_1)
	#print(phi_2)

	if (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and e2[i_2][1] > v[1]):	
		if phi_1 < phi_2:
			angle = phi_2 - phi_1
		else:
			angle = 360.0 - phi_1 + phi_2
	elif (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and e2[i_2][1] < v[1])\
	or (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and e2[i_2][1] > v[1]):
		angle = phi_2
	elif (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] > v[1]):
		angle = 360.0 - phi_1
	elif (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps)\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps):
		angle = 90.0 - phi_1
	elif (e1[i_1][0] > v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] > v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] < v[0] and e2[i_2][1] > v[1]):
		angle = 270.0 + phi_2
	elif (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps)\
	or (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps):
		angle = 90.0
	elif (e1[i_1][0] > v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] > v[1]):
		angle = 270.0
	elif (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and e2[i_2][1] > v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and e2[i_2][1] < v[1]):
		if phi_1 > phi_2:
			angle = phi_1 - phi_2
		else:
			angle = 360.0 - phi_2 + phi_1
	elif (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] > v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] < v[1]):
		angle = phi_1
	elif (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and e2[i_2][1] > v[1])\
	or (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and e2[i_2][1] < v[1]):
		angle = 360.0 - phi_2
	elif (e1[i_1][0] > v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] > v[0] and e2[i_2][1] > v[1])\
	or (e1[i_1][0] < v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] < v[0] and e2[i_2][1] < v[1]):
		angle = 90.0 - phi_2
	elif (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps)\
	or (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps):
		angle = 270.0 + phi_1
	elif (e1[i_1][0] > v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] > v[1])\
	or (e1[i_1][0] < v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] < v[1]):
		angle = 90.0
	elif (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps)\
	or (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps):
		angle = 270.0
	elif (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and e2[i_2][1] > v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and e2[i_2][1] < v[1]):
		angle = 180.0 - phi_1 - phi_2
	elif (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and e2[i_2][1] > v[1]):
		angle = 180.0 + phi_1 + phi_2
	elif (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and e2[i_2][1] > v[1])\
	or (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and e2[i_2][1] < v[1]):
		angle = 180.0 - phi_2
	elif (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] > v[1]):
		angle = 180.0 + phi_1
	elif (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] > v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] < v[1]):
		angle = 180.0 - phi_1
	elif (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and e2[i_2][1] < v[1])\
	or (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and e2[i_2][1] > v[1]):
		angle = 180.0 + phi_2
	elif (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] < v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] > v[1])\
	or (abs(e1[i_1][0] - v[0]) < np.finfo(float).eps and e1[i_1][1] > v[1] and abs(e2[i_2][0] - v[0]) < np.finfo(float).eps and e2[i_2][1] < v[1])\
	or (e1[i_1][0] > v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] < v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps)\
	or (e1[i_1][0] < v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] > v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps):
		angle = 180.0
	elif (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and e2[i_2][1] > v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and e2[i_2][1] < v[1]):
		angle = 180.0 - phi_1 + phi_2
	elif (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps)\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps):
		angle = 270.0 - phi_1
	elif (e1[i_1][0] < v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] > v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] > v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] < v[0] and e2[i_2][1] > v[1]):
		angle = 90.0 + phi_2
	elif (e1[i_1][0] > v[0] and e1[i_1][1] < v[1] and e2[i_2][0] < v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] > v[1] and e2[i_2][0] > v[0] and e2[i_2][1] > v[1]):
		angle = 360.0 - phi_1 - phi_2
	elif (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and e2[i_2][1] > v[1]):
		angle = phi_1 + phi_2
	elif (e1[i_1][0] > v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] < v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and abs(e1[i_1][1] - v[1]) < np.finfo(float).eps and e2[i_2][0] > v[0] and e2[i_2][1] > v[1]):
		angle = 270.0 - phi_2
	elif (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps)\
	or (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and abs(e2[i_2][1] - v[1]) < np.finfo(float).eps):
		angle = 90.0 + phi_1
	elif (e1[i_1][0] > v[0] and e1[i_1][1] > v[1] and e2[i_2][0] < v[0] and e2[i_2][1] < v[1])\
	or (e1[i_1][0] < v[0] and e1[i_1][1] < v[1] and e2[i_2][0] > v[0] and e2[i_2][1] > v[1]):
		angle = 180.0 - phi_2 + phi_1
	


	return angle



def createFan(V_domain_var, E_domain_var):
	E_new_temp = []
	E_new_temp_indices = []
	#n_vertices = 0
	#j_start = 0
	indices_i = []
	i_pred = 0
	i_succ = 0

	indices = ccwIndexList(V_domain_var)
		
	for i in range(0, len(indices)):
		
		if i == 0:
			indices_i = indices[2:(len(indices)-1)]
			i_pred = len(indices) - 1
			i_succ = i + 1
		elif i == len(indices) - 1:
			indices_i = []
			i_pred = i - 1
			i_succ = 0
		else:
			indices_i = indices[(i+2):len(indices)]
			i_pred = i - 1
			i_succ = i + 1

		
		for j in indices_i:
			e = (V_domain_var[indices[i]], V_domain_var[j])

			phi_d = getAngle((V_domain_var[indices[i_succ]], V_domain_var[indices[i]]), (V_domain_var[indices[i]], V_domain_var[indices[i_pred]]))
			phi_n = getAngle((V_domain_var[indices[i_succ]], V_domain_var[indices[i]]), e)
			
			if  phi_n < phi_d:
				if (not checkIntersection(e, E_domain_var)) and (not checkIntersection(e, E_new_temp)):
					E_new_temp.append(e)
					E_new_temp_indices.append((indices[i], j))

		
	#return E_new_temp
	return E_new_temp_indices




def getTriangulation(domain_var, E_new_var):
	T_temp = []

	indices = ccwIndexList(domain_var.getVBoundary())
	
	
	for e in E_new_var:
		#find triangles with 2 domain edges and one new edge
		d = domain_var.vrange(e[0], e[0]+2, False)
		d.__next__()
		p = d.__next__()
		
		d_rev = domain_var.vrange(e[0], e[0]+2, True)
		d_rev.__next__()
		q = d_rev.__next__()
		
		if d.__next__() == e[1]:
			triangle_vertices = [indices.index(e[0]), indices.index(p), indices.index(e[1])]
			triangle_vertices.sort()					
			T_temp.append((indices[triangle_vertices[0]], indices[triangle_vertices[1]], indices[triangle_vertices[2]]))
		
		if d_rev.__next__() == e[1]:
			triangle_vertices = [indices.index(e[0]), indices.index(q), indices.index(e[1])]
			triangle_vertices.sort()			
			T_temp.append((indices[triangle_vertices[0]], indices[triangle_vertices[1]], indices[triangle_vertices[2]]))
		
		#find triangles with 3 new edges
		e0_connected = list(filter(lambda x: (x[0] == e[0]) or (x[1] == e[0]), E_new_var))
		e1_connected = list(filter(lambda x: (x[0] == e[1]) or (x[1] == e[1]), E_new_var))
		
		p_list = list(set([x[0] if x[0] != e[0] else x[1] for x in e0_connected]) & set([x[0] if x[0] != e[1] else x[1] for x in e1_connected]))
		
		if len(p_list) != 0:
			t = [e[0], p_list[0], e[1]]
			triangle_vertices = [indices.index(e[0]), indices.index(p_list[0]), indices.index(e[1])]
			triangle_vertices.sort()
			t = [indices[triangle_vertices[0]], indices[triangle_vertices[1]], indices[triangle_vertices[2]]]
			
			if not (set(itertools.permutations(t)) & set(T_temp)):
				T_temp.append(tuple(t))

	#find triangles with 1 domain edge and 2 new edges
	for e in domain_var.getEBoundaryIndices():
		e0_connected = list(filter(lambda x: (x[0] == e[0]) or (x[1] == e[0]), E_new_var))
		e1_connected = list(filter(lambda x: (x[0] == e[1]) or (x[1] == e[1]), E_new_var))
		
		p_list = list(set([x[0] if x[0] != e[0] else x[1] for x in e0_connected]) & set([x[0] if x[0] != e[1] else x[1] for x in e1_connected]))
		if len(p_list) != 0:
			triangle_vertices = [indices.index(e[0]), indices.index(p_list[0]), indices.index(e[1])]
			triangle_vertices.sort()
			#T_temp.append((e[0], p_list[0], e[1]))
			T_temp.append((indices[triangle_vertices[0]], indices[triangle_vertices[1]], indices[triangle_vertices[2]]))
	
	
	angles_T = {}
	i_key = 0
	for t in T_temp:
		angles_T[i_key] = (getAngle((domain_var.getVBoundary()[t[0]], domain_var.getVBoundary()[t[1]]), (domain_var.getVBoundary()[t[0]], domain_var.getVBoundary()[t[2]])), getAngle((domain_var.getVBoundary()[t[1]], domain_var.getVBoundary()[t[2]]), (domain_var.getVBoundary()[t[1]], domain_var.getVBoundary()[t[0]])), getAngle((domain_var.getVBoundary()[t[2]], domain_var.getVBoundary()[t[0]]), (domain_var.getVBoundary()[t[2]], domain_var.getVBoundary()[t[1]])))
		i_key += 1

	#print(angles_T)

	angle_min_global_old = 180.0
	angles_T_ranked = {}
	for k,v in sorted(angles_T.items(), key = lambda x: min(x[1])):
		angles_T_ranked[k] = v
		if angle_min_global_old > min(v):
			angle_min_global_old = min(v)
	print(angles_T_ranked)
	
	
	print(T_temp)
	
	i_loop_test = 0 #angles_T durchlaufen
	angle_min_global_new = 0.0
	while(i_loop_test < len(list(angles_T_ranked.keys()))):

		print("i_loop_test: ", i_loop_test)

		#print(T_temp[list(angles_T_ranked.keys())[i_loop_test]])
		
		i_triangle = list(angles_T_ranked.keys())[i_loop_test]
		
		#find longest edge of triangle
		e_max_indices = ((0,0), (0,0))	
		if angles_T_ranked[i_triangle][0] == max(angles_T_ranked[i_triangle]):
			e_max_indices = (T_temp[i_triangle][1], T_temp[i_triangle][2])
		elif angles_T_ranked[i_triangle][1] == max(angles_T_ranked[i_triangle]):
			e_max_indices = (T_temp[i_triangle][2], T_temp[i_triangle][0])
		else:		
			e_max_indices = (T_temp[i_triangle][0], T_temp[i_triangle][1])

		#print(e_max_indices)


		for i in range(0, len(T_temp)):
			if (i != i_triangle) and (set(e_max_indices) < set(T_temp[i])):
				q = list(set(T_temp[i]) - set(e_max_indices))[0]			
				
				#check, whether e_max_indices is the longest edge of the adjacent triangle
				if (calculateEuclideanNorm2D(tuple(np.subtract(domain_var.getVBoundary()[e_max_indices[0]], domain_var.getVBoundary()[e_max_indices[1]]))) >= calculateEuclideanNorm2D(tuple(np.subtract(domain_var.getVBoundary()[e_max_indices[0]], domain_var.getVBoundary()[q])))) and (calculateEuclideanNorm2D(tuple(np.subtract(domain_var.getVBoundary()[e_max_indices[0]], domain_var.getVBoundary()[e_max_indices[1]]))) >= calculateEuclideanNorm2D(tuple(np.subtract(domain_var.getVBoundary()[e_max_indices[1]], domain_var.getVBoundary()[q])))):
					if np.linalg.det(np.array([[domain_var.getVBoundary()[T_temp[i_triangle][0]][0], domain_var.getVBoundary()[T_temp[i_triangle][0]][1], pow(calculateEuclideanNorm2D(domain_var.getVBoundary()[T_temp[i_triangle][0]]), 2), 1.0], [domain_var.getVBoundary()[T_temp[i_triangle][1]][0], domain_var.getVBoundary()[T_temp[i_triangle][1]][1], pow(calculateEuclideanNorm2D(domain_var.getVBoundary()[T_temp[i_triangle][1]]), 2), 1.0], [domain_var.getVBoundary()[T_temp[i_triangle][2]][0], domain_var.getVBoundary()[T_temp[i_triangle][2]][1], pow(calculateEuclideanNorm2D(domain_var.getVBoundary()[T_temp[i_triangle][2]]), 2), 1.0], [domain_var.getVBoundary()[q][0], domain_var.getVBoundary()[q][1], pow(calculateEuclideanNorm2D(domain_var.getVBoundary()[q]), 2), 1.0]])) > 0.0:
						#replace old edge with new edge						
						i_edge = E_new_var.index(*[x for x in E_new_var if set(x) == set(e_max_indices)])
						edge_new = (*(set(T_temp[i_triangle]) - set(e_max_indices)), q)
						
						#Edge flip: replace old triangles by new triangles (created by replaced edge)
						triangle_vertices = [indices.index(*(set(T_temp[i_triangle]) - set(e_max_indices))), indices.index(q), indices.index(e_max_indices[0])]
						triangle_vertices.sort()
						t_new_1 = (indices[triangle_vertices[0]], indices[triangle_vertices[1]], indices[triangle_vertices[2]])
						
						triangle_vertices = [indices.index(*(set(T_temp[i_triangle]) - set(e_max_indices))), indices.index(q), indices.index(e_max_indices[1])]
						triangle_vertices.sort()
						t_new_2 = (indices[triangle_vertices[0]], indices[triangle_vertices[1]], indices[triangle_vertices[2]])
						
						#calculate angles of new triangles and replace the values in angles_T_ranked
						#angles_t_new_1 = (getAngle((domain_var.getVBoundary()[T_temp[i][0]], domain_var.getVBoundary()[T_temp[i][1]]), (domain_var.getVBoundary()[T_temp[i][0]], domain_var.getVBoundary()[T_temp[i][2]])), getAngle((domain_var.getVBoundary()[T_temp[i][1]], domain_var.getVBoundary()[T_temp[i][2]]), (domain_var.getVBoundary()[T_temp[i][1]], domain_var.getVBoundary()[T_temp[i][0]])), getAngle((domain_var.getVBoundary()[T_temp[i][2]], domain_var.getVBoundary()[T_temp[i][0]]), (domain_var.getVBoundary()[T_temp[i][2]], domain_var.getVBoundary()[T_temp[i][1]])))						
						#angles_t_new_2 = (getAngle((domain_var.getVBoundary()[T_temp[i_triangle][0]], domain_var.getVBoundary()[T_temp[i_triangle][1]]), (domain_var.getVBoundary()[T_temp[i_triangle][0]], domain_var.getVBoundary()[T_temp[i_triangle][2]])), getAngle((domain_var.getVBoundary()[T_temp[i_triangle][1]], domain_var.getVBoundary()[T_temp[i_triangle][2]]), (domain_var.getVBoundary()[T_temp[i_triangle][1]], domain_var.getVBoundary()[T_temp[i_triangle][0]])), getAngle((domain_var.getVBoundary()[T_temp[i_triangle][2]], domain_var.getVBoundary()[T_temp[i_triangle][0]]), (domain_var.getVBoundary()[T_temp[i_triangle][2]], domain_var.getVBoundary()[T_temp[i_triangle][1]])))
						angles_t_new_1 = (getAngle((domain_var.getVBoundary()[t_new_1[0]], domain_var.getVBoundary()[t_new_1[1]]), (domain_var.getVBoundary()[t_new_1[0]], domain_var.getVBoundary()[t_new_1[2]])), getAngle((domain_var.getVBoundary()[t_new_1[1]], domain_var.getVBoundary()[t_new_1[2]]), (domain_var.getVBoundary()[t_new_1[1]], domain_var.getVBoundary()[t_new_1[0]])), getAngle((domain_var.getVBoundary()[t_new_1[2]], domain_var.getVBoundary()[t_new_1[0]]), (domain_var.getVBoundary()[t_new_1[2]], domain_var.getVBoundary()[t_new_1[1]])))						
						angles_t_new_2 = (getAngle((domain_var.getVBoundary()[t_new_2[0]], domain_var.getVBoundary()[t_new_2[1]]), (domain_var.getVBoundary()[t_new_2[0]], domain_var.getVBoundary()[t_new_2[2]])), getAngle((domain_var.getVBoundary()[t_new_2[1]], domain_var.getVBoundary()[t_new_2[2]]), (domain_var.getVBoundary()[t_new_2[1]], domain_var.getVBoundary()[t_new_2[0]])), getAngle((domain_var.getVBoundary()[t_new_2[2]], domain_var.getVBoundary()[t_new_2[0]]), (domain_var.getVBoundary()[t_new_2[2]], domain_var.getVBoundary()[t_new_2[1]])))

						angle_min_global_new = min(min(angles_t_new_1), min(angles_t_new_2))
						
						print("  T_temp: ", T_temp)
						print("  angles_T_ranked (keys): ", angles_T_ranked.keys())
						print("  i_triangle: ", i_triangle)
						print("  current triangle: ", T_temp[i_triangle])
						print("  old edge: ", e_max_indices)
						print("  new edge: ", edge_new)
						print("  new triangles: ", t_new_1, ", ", t_new_2)
						print("  angles_t_new_1: ", angles_t_new_1)
						print("  angles_t_new_2: ", angles_t_new_2)
						print("  angles: ", angles_T_ranked.values())
						print("  angle_min (old): ", angle_min_global_old, ", ", "angle_min (new): ", angle_min_global_new)

						if angle_min_global_new > angle_min_global_old:
							E_new_var[i_edge] = edge_new
							T_temp[i] = t_new_1
							T_temp[i_triangle] = t_new_2
							angles_T_ranked[i] = angles_t_new_1
							angles_T_ranked[i_triangle] = angles_t_new_2
							angles_T_ranked_temp = dict(sorted(angles_T_ranked.items(), key = lambda x: min(x[1])))
							angles_T_ranked = angles_T_ranked_temp.copy()
							angle_min_global_old = min(list(angles_T_ranked.values())[0])

							i_loop_test = -1
							
						

						#print("i =  ", i)
						#print("i_triangle = ", i_triangle)
						#print("angles_T_ranked: \n", angles_T_ranked)
						#print("angles_T: \n", angles_T)
						#del angles_T[i]
						#del angles_T[i_triangle]
						
		i_loop_test += 1

		
		#TODO: überprüfe in äußerer Schleife, ob tatsächlich eine bessere Triangulierung entsteht
		#TODO: dazu muss eine Funktion zur Bestimmung des minimalen Winkels der gesamten Triangulierung implementiert werden

	
	
#delaunay.pdf imrtalk.pdf

	return T_temp, E_new_var




def calculateEuclideanNorm2D(v_var):
	return math.sqrt(pow(v_var[0], 2) + pow(v_var[1], 2))



def ccwIndexList(V_domain_var): 
	x_max = V_domain_var[0][0]
	i_max = 0
	i_pred = 0
	i_succ = 0
	i_next = 0

	#find tuple with maximum x-value and store value and index
	for i in range(0, len(V_domain_var)-1):
		if x_max < V_domain_var[i][0]:
			x_max = V_domain_var[i][0]
			i_max = i
	
	
	if i_max == 0:
		i_pred = len(V_domain_var) - 1
	else:
		i_pred = i_max - 1

	if i_max == (len(V_domain_var) - 1):
		i_succ = 0
	else:
		i_succ = i_max + 1

	if (V_domain_var[i_pred][1] >= V_domain_var[i_max][1]) and (V_domain_var[i_succ][1] >= V_domain_var[i_max][1]):
		if (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (V_domain_var[i_pred][1] > V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (V_domain_var[i_succ][1] > V_domain_var[i_max][1]):
			phi_pred = 180.0/math.pi * np.arccos(abs(V_domain_var[i_pred][1] - V_domain_var[i_max][1])/(calculateEuclideanNorm2D(tuple(np.subtract(V_domain_var[i_pred], V_domain_var[i_max])))))
			phi_succ = 180.0/math.pi * np.arccos(abs(V_domain_var[i_succ][1] - V_domain_var[i_max][1])/(calculateEuclideanNorm2D(tuple(np.subtract(V_domain_var[i_succ], V_domain_var[i_max])))))
			
			if phi_pred < phi_succ:
				i_next = i_pred
			else:
				i_next = i_succ
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_pred][1] - V_domain_var[i_max][1]) < np.finfo(float).eps) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (V_domain_var[i_succ][1] > V_domain_var[i_max][1]):
			i_next = i_succ
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (V_domain_var[i_pred][1] > V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_succ][1] - V_domain_var[i_max][1]) < np.finfo(float).eps):
			i_next = i_pred
		elif (abs(V_domain_var[i_pred][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_pred][1] > V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (V_domain_var[i_succ][1] > V_domain_var[i_max][1]):
			i_next = i_pred
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (V_domain_var[i_pred][1] > V_domain_var[i_max][1]) and (abs(V_domain_var[i_succ][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_succ][1] > V_domain_var[i_max][1]):
			i_next = i_succ
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_pred][1] - V_domain_var[i_max][1]) < np.finfo(float).eps) and (abs(V_domain_var[i_succ][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_succ][1] > V_domain_var[i_max][1]):
			i_next = i_succ
		elif (abs(V_domain_var[i_pred][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_pred][1] > V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_succ][1] - V_domain_var[i_max][1]) < np.finfo(float).eps):
			i_next = i_pred
	elif (V_domain_var[i_pred][1] <= V_domain_var[i_max][1]) and (V_domain_var[i_succ][1] <= V_domain_var[i_max][1]):
		if (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (V_domain_var[i_pred][1] < V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (V_domain_var[i_succ][1] < V_domain_var[i_max][1]):

			phi_pred = 180.0/math.pi * np.arccos(abs(V_domain_var[i_pred][1] - V_domain_var[i_max][1])/(calculateEuclideanNorm2D(tuple(np.subtract(V_domain_var[i_pred], V_domain_var[i_max])))))
			phi_succ = 180.0/math.pi * np.arccos(abs(V_domain_var[i_succ][1] - V_domain_var[i_max][1])/(calculateEuclideanNorm2D(tuple(np.subtract(V_domain_var[i_succ], V_domain_var[i_max])))))
			
			if phi_pred > phi_succ:
				i_next = i_pred
			else:
				i_next = i_succ
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_pred][1] - V_domain_var[i_max][1]) < np.finfo(float).eps) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (V_domain_var[i_succ][1] < V_domain_var[i_max][1]):
			i_next = i_pred
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (V_domain_var[i_pred][1] < V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_succ][1] - V_domain_var[i_max][1]) < np.finfo(float).eps):
			i_next = i_succ
		elif (abs(V_domain_var[i_pred][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_pred][1] < V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (V_domain_var[i_succ][1] < V_domain_var[i_max][1]):
			i_next = i_succ
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (V_domain_var[i_pred][1] < V_domain_var[i_max][1]) and (abs(V_domain_var[i_succ][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_succ][1] < V_domain_var[i_max][1]):
			i_next = i_pred
		elif (V_domain_var[i_pred][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_pred][1] - V_domain_var[i_max][1]) < np.finfo(float).eps) and (abs(V_domain_var[i_succ][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_succ][1] < V_domain_var[i_max][1]):
			i_next = i_pred
		elif (abs(V_domain_var[i_pred][0] - V_domain_var[i_max][0]) < np.finfo(float).eps) and (V_domain_var[i_pred][1] < V_domain_var[i_max][1]) and (V_domain_var[i_succ][0] < V_domain_var[i_max][0]) and (abs(V_domain_var[i_succ][1] - V_domain_var[i_max][1]) < np.finfo(float).eps):
			i_next = i_succ
	else:
		if V_domain_var[i_pred][1] > V_domain_var[i_max][1]:
			i_next = i_pred
		else:
			i_next = i_succ
	
	
	
	indices = []	
	if i_max == 0:
		if i_next == len(V_domain_var)-1:			
			indices.append(i_max)
			indices = indices + [*range(len(V_domain_var)-1, i_max,-1)]			
		else:
			indices = [*range(i_max, len(V_domain_var))]
	elif i_max == (len(V_domain_var) - 1):
		if i_next == 0:
			indices.append(len(V_domain_var) - 1)
			indices = indices + [*range(0, len(V_domain_var) - 1)]
		else:
			indices = indices + [*range(len(V_domain_var) - 1, 0,-1)]
			indices.append(0)
	else:
		if i_next > i_max:
			indices = [*range(i_max, len(V_domain_var))]
			indices = indices + [*range(0, i_max)]
		else:
			indices = [*range(i_max, 0,-1)]
			indices.append(0)
			indices = indices + [*range(len(V_domain_var)-1, i_max,-1)]
	
	
	#print(indices)	

	return indices



def calculateNormalVectors(V_domain_var):
	normal_vectors_temp = {}
		
	indices = ccwIndexList(V_domain_var)
	
	
	for i in range(0, len(indices)):
		if i == 0:
			normal_vectors_temp[indices[i]] = [(V_domain_var[indices[i]][1] - V_domain_var[indices[len(indices)-1]][1], -(V_domain_var[indices[i]][0] - V_domain_var[indices[len(indices)-1]][0])), (V_domain_var[indices[i+1]][1] - V_domain_var[indices[i]][1], -(V_domain_var[indices[i+1]][0] - V_domain_var[indices[i]][0]))]
		elif i == len(indices) - 1:
			normal_vectors_temp[indices[i]] = [(V_domain_var[indices[i]][1] - V_domain_var[indices[i-1]][1], -(V_domain_var[indices[i]][0] - V_domain_var[indices[i-1]][0])), (V_domain_var[indices[0]][1] - V_domain_var[indices[i]][1], -(V_domain_var[indices[0]][0] - V_domain_var[indices[i]][0]))]
		else:
			normal_vectors_temp[indices[i]] = [(V_domain_var[indices[i]][1] - V_domain_var[indices[i-1]][1], -(V_domain_var[indices[i]][0] - V_domain_var[indices[i-1]][0])), (V_domain_var[indices[i+1]][1] - V_domain_var[indices[i]][1], -(V_domain_var[indices[i+1]][0] - V_domain_var[indices[i]][0]))]
	
	return normal_vectors_temp



#Example 1:
V_domain = [(-8.0, 6.0), (-6.0, 7.0), (2.0, 9.0), (7.0, 5.0), (6.0, -5.0), (1.0, -6.0), (-3.0, -7.0), (-5.0, -7.0), (-5.0, -3.0), (-8.0, -2.0), (-5.0, 2.0)]

#Example 2:
#V_domain = [(-4.0, 6.0), (2.0, 7.0), (5.0, 4.0), (2.0, -3.0), (4.0, -5.0), (7.0, -3.0), (7.0, -7.0), (-3.0, -7.0), (-7.0, -1.0)]

#TEST:
#V_domain = [(-8.0, 6.0), (-1.0, 6.0), (-1.0, 9.0), (-8.0, 9.0)]
#V_domain = [(-3.0, 6.0), (-1.0, 6.0), (-1.0, 9.0), (-3.0, 9.0)]

#V_domain = [(-1.0, 0.0), (1.0, 0.0), (2.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
#V_domain = [(-1.0, 1.0), (1.0, 1.0), (2.0, -1.0), (1.0, 0.0), (-1.0, 0.0)]
#V_domain = [(-1.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (-1.0, 1.0)]
#V_domain = [(-1.0, 1.0), (2.0, 1.0), (2.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]
#V_domain = [(-1.0, 0.0), (1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 1.0), (-1.0, 1.0)]
#V_domain = [(-1.0, 0.0), (0.0, 0.0), (0.0, 0.5), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (-1.0, 1.0)]


#V_domain = [(-1.0, 0.0), (1.0, 0.0), (2.0, 2.0), (1.0, 1.0), (-1.0, 1.0)]
#V_domain = [(-1.0, 1.0), (1.0, 1.0), (2.0, 2.0), (1.0, 0.0), (-1.0, 0.0)]
#V_domain = [(-1.0, 1.0), (1.0, 1.0), (2.0, 1.0), (1.0, 0.0), (-1.0, 0.0)]
#V_domain = [(-1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 0.5), (0.0, 1.0), (-1.0, 1.0)]
#V_domain = [(-1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.0, 1.0), (-1.0, 1.0)]

#V_domain = [(-1.0, 0.0), (1.0, 0.0), (2.0, 0.5), (1.0, 1.0), (-1.0, 1.0)]
#V_domain = [(-1.0, 1.0), (1.0, 1.0), (2.0, 0.5), (1.0, 0.0), (-1.0, 0.0)]
#V_domain = [(2.0, 0.5), (1.0, 0.0), (-1.0, 0.0), (-1.0, 1.0), (1.0, 1.0)]
#V_domain = [(-1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (1.0, 1.0), (-1.0, 1.0)]
#V_domain = [(-1.0, 1.0), (1.0, 1.0), (2.0, 0.5), (2.0, 0.0), (-1.0, 0.0)]
#V_domain = [(-1.0, 0.0), (1.0, 0.0), (2.0, 0.5), (2.0, 1.0), (-1.0, 1.0)]



domain = Domain(V_domain)

print(domain.getVBoundary())
print(domain.getEBoundary())

E_new = createFan(domain.getVBoundary(), domain.getEBoundary())
for k in range(0, len(E_new)):
	print(k,": ", E_new[k], ", (", domain.getVBoundary()[E_new[k][0]], ",", domain.getVBoundary()[E_new[k][1]],")")

T = []
T, E_new = getTriangulation(domain, E_new)
print("T: ", T)
print("E_new:", E_new)




main_window = Tk(className = "Triangulation")
main_window.geometry("600x600")

main_window.update_idletasks()
canvas = Canvas(main_window, width=main_window.winfo_width(), height= main_window.winfo_height())
canvas.grid(row=0, column=0, sticky=E+W+N+S)

#draw domain edges
for i in range(0, len(V_domain)-1):
	canvas.create_line(transformCoord(main_window, domain.getVBoundary(), domain.getVBoundary()[i]), transformCoord(main_window, domain.getVBoundary(), domain.getVBoundary()[i+1]))

canvas.create_line(transformCoord(main_window, domain.getVBoundary(), domain.getVBoundary()[len(domain.getVBoundary())-1]), transformCoord(main_window, domain.getVBoundary(), domain.getVBoundary()[0]))

#draw new edges
for i in range(0, len(E_new)):
	canvas.create_line(transformCoord(main_window, V_domain, V_domain[E_new[i][0]]), transformCoord(main_window, domain.getVBoundary(), domain.getVBoundary()[E_new[i][1]]), fill="red")



main_window.mainloop()
