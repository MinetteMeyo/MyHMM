# MyHMM
Source Code of my Decoding Sequence with Hidden Markov Models
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib . pyplot as plt
# % matplotlib inline
# create state space and initial state probabilities
states = [ ' o1 ' , ' o2 ' , ' o3 ']
hidden_states = [ ' congested ' , ' mixted ' , ' free ']
# observations = ( ' o1 ', ' o2 ', ' o3 ')
pi = [ 0 . 30 , 0 . 42 , 0 . 28 ]
state_space = pd . Series ( pi , index = hidden_states , name = ' states ')
print ( state_space )
print ( '\ n ' , state_space . sum () )
# create hidden transition matrix
# a or alpha
#
= transition probability matrix of changing states given a
state
# matrix is size ( M x M ) where M is number of states
a_df = pd . DataFrame ( columns = hidden_states , index = hidden_states )
a_df . loc [ hidden_states [ 0 ] ] = [ 0 . 55 , 0 . 35 , 0 . 1 ]
a_df . loc [ hidden_states [ 1 ] ] = [ 0 . 35 , 0 .4 , 0 . 25 ]
a_df . loc [ hidden_states [ 2 ] ] = [ 0 .2 , 0 .5 , 0 . 3 ]
print ( a_df )
a = a_df . values
print ( '\ n ' , a , a . shape , '\ n ')
print ( a_df . sum ( axis = 1 ) )
# create matrix of observation ( emission ) probabilities
# b or beta = observation probabilities given state
# matrix is size ( M x O ) where M is number of states
# and O is number of different possible observations
observable_states = states
b_df = pd . DataFrame ( columns = observable_states , index = hidden_states
)
b_df . loc [ hidden_states [ 0 ] ] = [ 0 . 71 , 0 . 21 , 0 . 08 ]
b_df . loc [ hidden_states [ 1 ] ] = [ 0 . 25 , 0 .6 , 0 . 15 ]
b_df . loc [ hidden_states [ 2 ] ] = [ 0 . 05 , 0 . 25 , 0 . 7 ]
print ( b_df )
b = b_df . values
print ( '\ n ' , b , b . shape , '\ n ')
print ( b_df . sum ( axis = 1 ) )
# observation sequence of the road traffic 's behaviors
# observations are encoded numerically
obs_map = { ' o1 ':0 , ' o2 ':1 , ' o3 ': 2 }
obs = np . array ( [2 ,1 ,0 ,1 , 1 ] )
inv_obs_map = dict (( v , k ) for k , v in obs_map . items () )
obs_seq = [ inv_obs_map [ v ] for v in list ( obs ) ]
print ( pd . DataFrame ( np . column_stack ( [ obs , obs_seq ] ) ,
columns = [ ' Obs_code ' , ' Obs_seq '] ) )
def viterbi ( pi , a , b , obs ) :
nStates = np . shape ( b ) [ 0 ]
T = np . shape ( obs ) [ 0 ]
# init blank path
path = np . zeros ( T )
# delta --> highest probability of any path that reaches state i
delta = np . zeros (( nStates , T ) )
# phi --> argmax by time step for each state
phi = np . zeros (( nStates , T ) )
# init delta and phi
delta [ : , 0 ] = pi * b [ : , obs [ 0 ] ]
phi [ : , 0 ] = 0
print ( '\ nStart Walk Forward \ n ')
# the forward algorithm extension
for t in range (1 , T ) :
for s in range ( nStates ) :
delta [s , t ] = np . max ( delta [ : , t - 1 ] * a [ : , s ] ) * b [s , obs [ t ] ]
phi [s , t ] = np . argmax ( delta [ : , t - 1 ] * a [ : , s ] )
print ( 's = { s } and t = { t } : phi [ { s } , { t } ] = { phi } '. format ( s =s , t
=t , phi = phi [s , t ] ) )
# find optimal path
print ( ' - '* 50 )
print ( ' Start Backtrace \ n ')
path [ T - 1 ] = np . argmax ( delta [ : , T - 1 ] )
# p ( ' init path \ n
t ={ } path [{ } -1 ]= { } \ n '. format (T -1 , T , path
[T - 1 ]) )
for t in range ( T -2 , -1 , - 1 ) :
# path [ t ] = phi [ int ( path [ t + 1 ]) , t + 1 ]
path [ t ] = phi [ int ( path [ t + 1 ] ) , [ t + 1 ] ]
# p ( ' '* 4 + 't ={ t } , path [{ t } + 1 ]= { path } , [ { t } + 1 ]= { i } '. format ( t =t
, path = path [ t + 1 ] , i =[ t + 1
]) )
print ( ' path [ { } ] = { } '. format (t , path [ t ] ) )
return path , delta , phi
path , delta , phi = viterbi ( pi , a , b , obs )
print ( '\ nsingle best state path : \ n ' , path )
print ( ' delta :\ n ' , delta )
print ( ' phi :\ n ' , phi )
state_map = { 0 : ' congested ' , 1 : ' mixted ' , 2 : ' free '}
state_path = [ state_map [ v ] for v in path ]
( pd . DataFrame ()
. assign ( Observation = obs_seq )
. assign ( Best_Path = state_path ) )
