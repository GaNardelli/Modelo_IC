#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4  2021

@author: Gabriel Nardelli
Obj: desenvolvimento e otimização do modelo de programação linear
"""

# Controle do tempo de execução

import time

# File log de saída

import sys
sys.stdout = open('log.txt', 'w')
startTime = time.time()
print("Tempo de início = %d" %startTime)

# Bibliotecas 
from sympy import pretty_print as pp, latex
import numpy as np
import matplotlib.pyplot as plt
# import scipy.io as sio
import scipy.sparse as sp
import h5py
# from scipy.interpolate import interp1d
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from docplex.mp.progress import *
# from docplex.util.environment import get_environment

def get_h5py_struct_array_value(f, structname, fieldname, idx):
    ref = f[structname][fieldname][idx,0]
    #val = f[ref].value
    val = f[ref][()]
    return val

mdl = Model()

def build_var_agulhas(agulhas):
    # Variáveis do modelo 
    t = mdl.continuous_var_list(agulhas, lb = 0, name='t')
    print("Variaveis de tempo criada")
    
    #mdl.add_sos2(t)
    return t

def build_var_custos(pontos, Name):
    c = mdl.continuous_var_list(pontos,name=Name, lb = 0)
    #c = mdl.binary_var_list(pontos,name=Name, lb = 0)
    print("Custos tecido %s: OK" %Name)
    return c

def func_obj(c,pontos):
    fo = 0
    vetor_iteracao = []
    k = 0
    for s in range(len(c)):
        for i in range(len(c[s])):
            fo = fo + (c[s][i]/pontos[s])
            vetor_iteracao.append(k+1)
    return fo

def func_obj_reduzida(c,pontos,k):
    fo = 0
    for s in range(len(c)):
        for i in range(k):
            fo = fo + (c[s][i]/pontos[s])
    return fo

def func_obj_tecido(c,pontos,s):
    fo = 0
    for i in range(pontos[s]):
        fo = fo + (c[s][i]/pontos[s])
    return fo

def resolve_tecido(Dma,Dmi,d,Mma,Mmi,t,c,s=0):
    '''
    Parameters
    ----------
    Dma : List
        Lista de dose máxima permitida.
    Dmi : List
        Lista de dose minima permitida.
    d : Matrix
        Matriz de todas as doses de todos os tecidos.
    Mma : List
        Penalidade maxima.
    Mmi : List
        Penalidade minima.
    t : List Variables
        Lista de variaveis de tempo de permanência das sementes.
    c : List
        Lista de variaveis de custo.
    s : Int
        Tecido desejado
    Returns
    -------
    None.

    '''
    for i in range(len(c[s])):
        soma_interna_min = mdl.sum(Mmi[s]*d[s][i,j]*t[j] for j in range(len(t)))
        soma_interna_max = mdl.sum(Mma[s]*d[s][i,j]*t[j] for j in range(len(t)))
        stmi = mdl.sum(c[s][i] + soma_interna_min)
        stma = mdl.sum(c[s][i] - soma_interna_max)
        mdl.add_constraint( stmi >= Mmi[s]*Dmi[s])
        mdl.add_constraint( stma >= -Mma[s]*Dma[s])  
            
    print("######### Modelo Para Um Tecido #########")
    # Função Objetivo
    fo = func_obj_tecido(c, pontos_full,s)
    print("Função Objetivo Criada")
    # Solução 
    mdl.minimize(fo)    
    solution = mdl.solve()
    '''
    print("Número de variáveis = ", mdl.number_of_variables)
    if solution:
        print("Tem solução")
        print(mdl.solve_details)
        print(mdl.print_solution())
    else:
        print("Não tem Solução")
    '''    
    return t, c, solution
    
def resolve_reduzido(Dma,Dmi,d,Mma,Mmi,t,c,k=5):
    '''
    Parameters
    ----------
    Dma : List
        Lista de dose máxima permitida.
    Dmi : List
        Lista de dose minima permitida.
    d : Matrix
        Matriz de todas as doses de todos os tecidos.
    Mma : List
        Penalidade maxima.
    Mmi : List
        Penalidade minima.
    t : List Variables
        Lista de variaveis de tempo de permanência das sementes.
    c : List
        Lista de variaveis de custo.
    i : Int
        Numero de pontos a serem analisado.
    Returns
    -------
    None.

    '''
    for s in range(len(c)):
        for i in range(k):
            soma_interna_min = mdl.sum(Mmi[s]*d[s][i,j]*t[j] for j in range(len(t)))
            soma_interna_max = mdl.sum(Mma[s]*d[s][i,j]*t[j] for j in range(len(t)))
            stmi = mdl.sum(c[s][i] + soma_interna_min)
            stma = mdl.sum(c[s][i] - soma_interna_max)
            mdl.add_constraint( stmi >= Mmi[s]*Dmi[s])
            mdl.add_constraint( stma >= -Mma[s]*Dma[s])  
            
    print("######### Modelo Reduzido #########")
    # Função Objetivo
    fo = func_obj_reduzida(c, pontos_full,k)
    print("Função Objetivo Criada")
    # Solução 
    mdl.minimize(fo)    
    solution = mdl.solve()
    print("Número de variáveis = ", mdl.number_of_variables)
    if solution:
        print("Tem solução")
        print(mdl.solve_details)
        print(mdl.print_solution())
    else:
        print("Não tem Solução")
    
    return t, c, solution

def build_BATD_exact(Dma,Dmi,d,Mma,Mmi,t,c):
    '''
     Parameters
    ----------
    Dma : List
        Lista de dose máxima permitida.
    Dmi : List
        Lista de dose minima permitida.
    d : Matrix
        Matriz de todas as doses de todos os tecidos.
    Mma : List
        Penalidade maxima.
    Mmi : List
        Penalidade minima.
    t : List Variables
        Lista de variaveis de tempo de permanência das sementes.
    c : List
        Lista de variaveis de custo.
        
    Returns
    -------
    Tempo, custo, solução

    '''
    for s in range(len(c)):
        for i in range(len(c[s])):
            soma_interna_min = mdl.sum(Mmi[s]*d[s][i,j]*t[j] for j in range(len(t)))
            soma_interna_max = mdl.sum(Mma[s]*d[s][i,j]*t[j] for j in range(len(t)))
            stmi = c[s][i] + soma_interna_min
            stma = c[s][i] - soma_interna_max
            mdl.add_constraint( stmi >= Mmi[s]*Dmi[s])
            mdl.add_constraint( stma >= -Mma[s]*Dma[s]) 
            
    print("######### Modelo Completo #########")
    # Função Objetivo
    fo = func_obj(c, pontos_full)
    print("Função Objetivo Criada")
    # Solução 
    mdl.minimize(fo)    
    solution = mdl.solve()
    print("Número de variáveis = ", mdl.number_of_variables)
    if solution:
        print("Tem solução")
        print(mdl.solve_details)
        print(mdl.print_solution())
    else:
        print("Não tem Solução")  
    
    return t, c, solution

def slack_variable(pontos):
   x = mdl.continuous_var_list(pontos, lb = 0)
   return x
            
            
def build_BATD_custo_alterado(Dma,Dmi,d,Mma,Mmi,t,c,slack_max,slack_min):
    '''
     Parameters
    ----------
    Dma : List
        Lista de dose máxima permitida.
    Dmi : List
        Lista de dose minima permitida.
    d : Matrix
        Matriz de todas as doses de todos os tecidos.
    Mma : List
        Penalidade maxima.
    Mmi : List
        Penalidade minima.
    t : List Variables
        Lista de variaveis de tempo de permanência das sementes.
    c : List
        Lista de variaveis de custo.
        
    Returns
    -------
    Tempo, custo, solução

    '''
    for s in range(len(c)):
        for i in range(len(c[s])):
            soma_interna_min = mdl.sum(Mmi[s]*d[s][i,j]*t[j] for j in range(len(t)))
            soma_interna_max = mdl.sum(Mma[s]*d[s][i,j]*t[j] for j in range(len(t)))
            stmi = c[s][i] + soma_interna_min - slack_min[s][i]
            stma = c[s][i] - soma_interna_max - slack_max[s][i]
            mdl.add_constraint( stmi == Mmi[s]*Dmi[s])
            mdl.add_constraint( stma == -Mma[s]*Dma[s]) 
    
    print("######### Modelo Completo #########")
    # Função Objetivo
    fo = func_obj(c, pontos_full)
    print("Função Objetivo Criada")
    # Solução 
    mdl.minimize(fo)    
    solution = mdl.solve()
    print("Número de variáveis = ", mdl.number_of_variables)
    if solution:
        print("Tem solução")
        print(mdl.solve_details)
        print(mdl.print_solution())
    else:
        print("Não tem Solução")  
    
    return t, c, solution
    
def plota_custo(custo):
    label_custo = []
    for i in range(len(custo)):
        label_custo.append([j for j in range(len(custo[i]))])  
    plt.scatter(label_custo[0], custo[0] , color = 'red', label = 'Uretra')
    plt.scatter(label_custo[1], custo[1], color='green', label = 'Reto')
    plt.scatter(label_custo[2], custo[2] , color='blue', label = 'Bexiga')
    plt.scatter(label_custo[3], custo[3], color='orange', label = 'Próstata')
    plt.scatter(label_custo[4], custo[4] , color='black', label = 'Bexiga Parcial')
    plt.scatter(label_custo[5], custo[5], color='yellow', label = 'Reto Parcial')
    plt.scatter(label_custo[6], custo[6], color='brown', label = 'Tecido Saudável')
    
    plt.xlabel('Agulha')
    plt.ylabel('Custo')
    plt.title('Custo $c_{si}$')
    plt.legend()
    plt.show()
    

def plota_dose(d,t):
    '''Função que calcula a dose depositada em cada ponto i em cada tecido s'''
    dose_total = []
    ponto_total = []
    for s in range(len(d)):
        ponto, agulha = d[s].shape
        ponto_total.append(ponto)
        soma = 0
        dose_tecido = []
        for i in range(ponto):
            for j in range(agulha):
                soma = soma + (d[s][i,j]*t[j]/60)
            dose_tecido.append(soma)
            soma = 0
        dose_total.append(dose_tecido)
        
    tecido = ['Uretra', 'Reto', 'Bexiga', 'Próstata', 'Bexiga Parcial',
              'Reto Parcial', 'Tecido Saudável']
    dose = []
    for i in range(len(d)):
        dose.append(sum(dose_total[i]))
        
    plt.bar(tecido, dose, color = 'red', width=1)
    plt.xticks(rotation = 45)
    
    '''
    label_ponto = [j for j in range(ponto_total[0])]
    plt.plot(label_ponto,dose_total[0], color='red', label = 'Uretra')
    label_ponto = [j for j in range(ponto_total[1])]
    plt.plot(label_ponto,dose_total[1], color='green', label = 'Reto')
    label_ponto = [j for j in range(ponto_total[2])]
    plt.plot(label_ponto,dose_total[2], color='blue', label = 'Bexiga')
    label_ponto = [j for j in range(ponto_total[3])]
    plt.plot(label_ponto,dose_total[3], color='orange', label = 'Próstata')
    label_ponto = [j for j in range(ponto_total[4])]
    plt.plot(label_ponto,dose_total[4], color='black', label = 'Bexiga Parcial')
    label_ponto = [j for j in range(ponto_total[5])]
    plt.plot(label_ponto,dose_total[5], color='yellow', label = 'Reto Parcial')
    label_ponto = [j for j in range(ponto_total[6])]
    plt.plot(label_ponto,dose_total[6], color='brown', label = 'Tecido Saudável')
    '''
    plt.xlabel('Tecido')
    plt.ylabel('Dose cGy')
    plt.title('Distribuição de dose por tecido - caso 25')
    plt.legend()
    plt.show()
    
    return dose

def plota_tempo(t):
    label_t = [j for j in range(agulhas)]
    plt.bar(label_t,t,color = 'blue')
    plt.xlabel('Semente')
    plt.ylabel('Tempo (s)')
    plt.title('Comparação do tempo de permanência de cada semente - caso 01')
    plt.legend()
    plt.show()

    
def compara_tempo(t,t_ca):
    label_t = [j for j in range(agulhas)]
    plt.bar(label_t,t,color = 'blue', label = 'Modelo Alterovitz')
    plt.bar(label_t,t_ca,color = 'red', label = 'Modelo Holm')
    plt.xlabel('Semente')
    plt.ylabel('Tempo (s)')
    plt.title('Comparação do tempo de permanência de cada semente - caso 01')
    plt.legend()
    plt.show()

def obj_moren(w,q,x,p,d,target):
    fo = 0
    soma3 = 0
    descarte, ponto_target = d[target].shape()
    soma1 = mdl.sum(w[i] for i in range(ponto_target))
    soma1 = p*soma1
    soma2 = mdl.sum(q)
    
def modelo_moren(d,t,p,q,L,U,M,target):
    w = 0
    x = 0
    for i in range(len(d[target])):
        if sum(d[target][i,j] for j in range(len(t))) < L:
            w += 1
            
    for s in range(len(d)):
        if sum(d[s][i,j] for j in range(len(t))) > U:   
            x += 1
            
    w = mdl.continuous_var_list(w)
    x = mdl.continuous_var_list(x)
    
    
    
    
    
# =============================================================================
# Parametros clinicos constantes
# =============================================================================
# Tecidos: Ure, Rec_part, Blad_part, prostate, bla, rec, tecido saudavel
Mmin = [30,0,0,30,0,0,0] # Penalidade por dose minima ultrapassada
Dmin = [23,0,0,38,0,0,0]  # Dose minima recebida
Mmax = [100,100,100,100,100,100,100]  # Penalidade por dose máxima ultrapassada
Dmax = [45,26.6,26.6,38,26.6,26.6,26.6] # Dose máxima permitida


MminA = [100,0,0,100,0,0,0] # Penalidade por dose minima ultrapassada
DminA = [950,0,0,950,0,0,0]  # Dose minima recebida
MmaxA = [30,30,30,100,30,30,30]  # Penalidade por dose máxima ultrapassada
DmaxA = [1145,950,950,1475,950,950,950] # Dose máxima permitida

Mminh = [30,100,0,0,0] # Penalidade por dose minima ultrapassada
Dminh = [25,24,0.266,0.266,0]  # Dose minima recebida
Mmaxh = [30,100,30,30,30]  # Penalidade por dose máxima ultrapassada
Dmaxh = [45.600,38,26.600,26.600,0] # Dose máxima permitida

s = 7 #numero de tecidos
# =============================================================================
# Leitura do arquivo de entrada
# =============================================================================

filename = '/home/gabriel/Facul/IC/modelo_ic/Data/Prostate_BT_01.mat' # Arquivo de entrada
f = h5py.File(filename)

ure = {'data':[], 'row_ind':[], 'col_ind':[]}
rec_part = {'data':[], 'row_ind':[], 'col_ind':[]} 
blad_part = {'data':[], 'row_ind':[], 'col_ind':[]}
prostate = {'data':[], 'row_ind':[], 'col_ind':[]} 
bla = {'data':[], 'row_ind':[], 'col_ind':[]}
rec = {'data':[], 'row_ind':[], 'col_ind':[]}
naoesp = {'data':[], 'row_ind':[], 'col_ind':[]}

row_offset = 0
num_aux_variables = 0

for i in range(len(f['problem']['dataID'])):
    data_id = int(get_h5py_struct_array_value(f, 'problem', 'dataID', i))
    is_constraint = bool(get_h5py_struct_array_value(f, 'problem', 'IsConstraint', i))
    minimize = bool(get_h5py_struct_array_value(f, 'problem', 'Minimise', i))
    weight = float(get_h5py_struct_array_value(f, 'problem', 'Weight', i))
    bound = float(get_h5py_struct_array_value(f, 'problem', 'Objective', i))
    
    factor = 1 if minimize else -1
    
    # create the (i,j,value) pairs of the nonzeros in the constraint matrix
    ref = f['data']['matrix']['A'][data_id-1,0]
    tecido = f['data']['matrix']['Name'][data_id-1,0]
  
    if f[ref].attrs.get('MATLAB_class') == b'double':
        if data_id == 1 and ure['data'][0]==[]:
            ure['data'].append(factor*f[ref]['data'][()])
            ure['row_ind'].append(row_offset + f[ref]['ir'][()])
            jc = f[ref]['jc'][()]
        
            # decompress compressed column indices
            col_ind = np.zeros(jc[-1], dtype=np.uint64)
            for j in range(len(jc)-1):
                for k in range(jc[j],jc[j+1]):
                    col_ind[k] = j
            ure['col_ind'].append(col_ind)
            
            del j, k
            
        if data_id == 2 and rec_part['data']==[]:
            rec_part['data'].append(factor*f[ref]['data'][()])
            rec_part['row_ind'].append(row_offset + f[ref]['ir'][()])
            jc = f[ref]['jc'][()]
            
            # decompress compressed column indices
            col_ind = np.zeros(jc[-1], dtype=np.uint64)
            for j in range(len(jc)-1):
                for k in range(jc[j],jc[j+1]):
                    col_ind[k] = j
            rec_part['col_ind'].append(col_ind)
            
            del j, k
        
        
        if data_id == 3 and blad_part['data']==[]:
            jc = f[ref]['jc'][()]
            blad_part['data'].append(factor*f[ref]['data'][()])
            blad_part['row_ind'].append(row_offset + f[ref]['ir'][()])
        
            # decompress compressed column indices
            col_ind = np.zeros(jc[-1], dtype=np.uint64)
            for j in range(len(jc)-1):
                for k in range(jc[j],jc[j+1]):
                    col_ind[k] = j
            blad_part['col_ind'].append(col_ind)
            
            del j, k
            
        if data_id == 4 and prostate['data']==[]:
            jc = f[ref]['jc'][()]
            prostate['data'].append(factor*f[ref]['data'][()])
            prostate['row_ind'].append(row_offset + f[ref]['ir'][()])
            
            # decompress compressed column indices
            col_ind = np.zeros(jc[-1], dtype=np.uint64)
            for j in range(len(jc)-1):
                for k in range(jc[j],jc[j+1]):
                    col_ind[k] = j
            prostate['col_ind'].append(col_ind)
            
            del j, k
            
        if data_id == 5 and bla['data']==[]:
            jc = f[ref]['jc'][()]
            bla['data'].append(factor*f[ref]['data'][()])
            bla['row_ind'].append(row_offset + f[ref]['ir'][()])
        
            # decompress compressed column indices
            col_ind = np.zeros(jc[-1], dtype=np.uint64)
            for j in range(len(jc)-1):
                for k in range(jc[j],jc[j+1]):
                    col_ind[k] = j
            ure['col_ind'].append(col_ind)
            
            del j, k
            
        if data_id == 6 and rec['data']==[]:
            jc = f[ref]['jc'][()]
            rec['data'].append(factor*f[ref]['data'][()])
            rec['row_ind'].append(row_offset + f[ref]['ir'][()])
            
            # decompress compressed column indices
            col_ind = np.zeros(jc[-1], dtype=np.uint64)
            for j in range(len(jc)-1):
                for k in range(jc[j],jc[j+1]):
                    col_ind[k] = j
            rec['col_ind'].append(col_ind)
            
            del j, k
       
        if data_id == 7 and rec['data']==[]:
            jc = f[ref]['jc'][()]
            naoesp['data'].append(factor*f[ref]['data'][()])
            naoesp['row_ind'].append(row_offset + f[ref]['ir'][()])
            
            # decompress compressed column indices
            col_ind = np.zeros(jc[-1], dtype=np.uint64)
            for j in range(len(jc)-1):
                for k in range(jc[j],jc[j+1]):
                    col_ind[k] = j
            naoesp['col_ind'].append(col_ind)
            
            del j, k
             
        #jc = f[ref]['jc'].value
        jc = f[ref]['jc'][()]
        
        num_voxels = int(f[ref].attrs.get('MATLAB_sparse'))
    
        num_pencil_beams = jc.size-1
        del jc
        
    else:
        # dose matrix is in dense format and is transposed
        #DT = f[ref].value
        DT = f[ref][()]
        
        (num_pencil_beams,num_voxels) = DT.shape
        (col_ind,row_ind) = np.nonzero(DT)
  
        if data_id == 1 and ure['data']==[]:
            ure['data'].append(factor*DT[col_ind, row_ind])
            ure['row_ind'].append(row_offset + row_ind)
            ure['col_ind'].append(col_ind)
            
        if data_id == 2 and rec_part['data']==[]:
            rec_part['data'].append(factor*DT[col_ind, row_ind])
            rec_part['row_ind'].append(row_offset + row_ind)
            rec_part['col_ind'].append(col_ind)
            
        if data_id == 3 and blad_part['data']==[]:
            blad_part['data'].append(factor*DT[col_ind, row_ind])
            blad_part['row_ind'].append(row_offset + row_ind)
            blad_part['col_ind'].append(col_ind)
            
        if data_id == 4 and prostate['data']==[]:
            prostate['data'].append(factor*DT[col_ind, row_ind])
            prostate['row_ind'].append(row_offset + row_ind)
            prostate['col_ind'].append(col_ind)
            
        if data_id == 5 and bla['data']==[]:
            bla['data'].append(factor*DT[col_ind, row_ind])
            bla['row_ind'].append(row_offset + row_ind)
            bla['col_ind'].append(col_ind)
            
        if data_id == 6 and rec['data']==[]:
            rec['data'].append(factor*DT[col_ind, row_ind])
            rec['row_ind'].append(row_offset + row_ind)
            rec['col_ind'].append(col_ind)
            
        if data_id == 7 and naoesp['data']==[]:
            naoesp['data'].append(factor*DT[col_ind, row_ind])
            naoesp['row_ind'].append(row_offset + row_ind)
            naoesp['col_ind'].append(col_ind)
            
            
        del DT, col_ind, row_ind
   
del f, i, data_id, is_constraint, minimize, weight, bound, factor, num_voxels        
# =============================================================================
# Matrizes de pontos de dose em cada tecido
# =============================================================================      

bla['data']     = np.concatenate(bla['data'])
bla['row_ind']  = np.concatenate(bla['row_ind'])
bla['col_ind']  = np.concatenate(bla['col_ind'])

blad_part['data']     = np.concatenate(blad_part['data'])
blad_part['row_ind']  = np.concatenate(blad_part['row_ind'])
blad_part['col_ind']  = np.concatenate(blad_part['col_ind'])

prostate['data']     = np.concatenate(prostate['data'])
prostate['row_ind']  = np.concatenate(prostate['row_ind'])
prostate['col_ind']  = np.concatenate(prostate['col_ind'])

rec['data']     = np.concatenate(rec['data'])
rec['row_ind']  = np.concatenate(rec['row_ind'])
rec['col_ind']  = np.concatenate(rec['col_ind'])

rec_part['data']     = np.concatenate(rec_part['data'])
rec_part['row_ind']  = np.concatenate(rec_part['row_ind'])
rec_part['col_ind']  = np.concatenate(rec_part['col_ind'])

ure['data']     = np.concatenate(ure['data'])
ure['row_ind']  = np.concatenate(ure['row_ind'])
ure['col_ind']  = np.concatenate(ure['col_ind'])

naoesp['data']     = np.concatenate(naoesp['data'])
naoesp['row_ind']  = np.concatenate(naoesp['row_ind'])
naoesp['col_ind']  = np.concatenate(naoesp['col_ind'])

ure_dense = sp.csr_matrix( (ure['data'],(ure['row_ind'],ure['col_ind'])) ).todense()
rec_part_dense = sp.csr_matrix( (rec_part['data'],(rec_part['row_ind'],rec_part['col_ind'])) ).todense()
blad_part_dense = sp.csr_matrix( (blad_part['data'],(blad_part['row_ind'],blad_part['col_ind'])) ).todense()
prostate_dense = sp.csr_matrix( (prostate['data'],(prostate['row_ind'],prostate['col_ind']))).todense()
blad_dense = sp.csr_matrix( (bla['data'],(bla['row_ind'],bla['col_ind']))).todense()
rec_dense = sp.csr_matrix( (rec['data'],(rec['row_ind'],rec['col_ind']))).todense()
naoesp_dense = sp.csr_matrix( (naoesp['data'],(naoesp['row_ind'],naoesp['col_ind']))).todense()
prostate_dense = prostate_dense*(-1)

del bla, blad_part, naoesp, prostate, rec,rec_part,ure

# Cria as variaveis de tempo
descarte, agulhas = ure_dense.shape
t = build_var_agulhas(agulhas)

# =============================================================================
# Calculo do numero total de pontos de dose para cada tecido
# =============================================================================

pontos_ure,descarte = ure_dense.shape
pontos_rec_part,descarte = rec_part_dense.shape
pontos_blad_part,descarte = blad_part_dense.shape
pontos_prostate,descarte = prostate_dense.shape
pontos_blad,descarte = blad_dense.shape
pontos_rec,descarte = rec_dense.shape
pontos_naoesp,descarte = naoesp_dense.shape

del descarte

pontos_full = []
pontos_full.append(pontos_ure)
pontos_full.append(pontos_rec_part)
pontos_full.append(pontos_blad_part)
pontos_full.append(pontos_prostate)
pontos_full.append(pontos_blad)
pontos_full.append(pontos_rec)
pontos_full.append(pontos_naoesp)

'''
# =============================================================================
# Retirando os pontos parciais versão teste
# =============================================================================
pontos_half = []
pontos_half.append(pontos_ure)
pontos_half.append(pontos_prostate)
pontos_half.append(pontos_blad)
pontos_half.append(pontos_rec)
pontos_half.append(pontos_naoesp)
'''

matriz_dose_full = []
matriz_dose_full.append(ure_dense)
matriz_dose_full.append(rec_part_dense)
matriz_dose_full.append(blad_part_dense)
matriz_dose_full.append(prostate_dense)
matriz_dose_full.append(blad_dense)
matriz_dose_full.append(rec_dense)
matriz_dose_full.append(naoesp_dense)

'''
# =============================================================================
# Matriz de dos de teste
# =============================================================================

matriz_dose_half = []
matriz_dose_half.append(ure_dense)
matriz_dose_half.append(prostate_dense)
matriz_dose_half.append(blad_dense)
matriz_dose_half.append(rec_dense)
matriz_dose_half.append(naoesp_dense)
'''
# Variaveis de Custo e slack
c = []
slack_max = []
slack_min = []

c_ure = build_var_custos(pontos_ure,"U") #uretra
slack_max_ure = slack_variable(pontos_ure)
slack_min_ure = slack_variable(pontos_ure)

c_rec_part = build_var_custos(pontos_rec_part,"RP") #reto parcial
slack_max_rec_part = slack_variable(pontos_rec_part)
slack_min_rec_part = slack_variable(pontos_rec_part)

c_blad_part = build_var_custos(pontos_blad_part, "BP") #bexiga parcial
slack_max_blad_part = slack_variable(pontos_blad_part)
slack_min_blad_part = slack_variable(pontos_blad_part) 

c_prostate = build_var_custos(pontos_prostate, "P") # prostata
slack_max_prostate = slack_variable(pontos_prostate)
slack_min_prostate = slack_variable(pontos_prostate)

c_blad = build_var_custos(pontos_blad, "B") #bexiga
slack_max_blad = slack_variable(pontos_blad)
slack_min_blad = slack_variable(pontos_blad)

c_rec = build_var_custos(pontos_rec, "R") #reto
slack_max_rec = slack_variable(pontos_rec)
slack_min_rec = slack_variable(pontos_rec) 

c_naoesp = build_var_custos(pontos_naoesp, "S") #saudavel
slack_max_naoesp = slack_variable(pontos_naoesp)
slack_min_naoesp = slack_variable(pontos_naoesp)

c.append(c_ure)
slack_max.append(slack_max_ure)
slack_min.append(slack_min_ure)

c.append(c_rec_part)
slack_max.append(slack_max_rec_part)
slack_min.append(slack_min_rec_part)

c.append(c_blad_part)
slack_max.append(slack_max_blad_part)
slack_min.append(slack_min_blad_part)

c.append(c_prostate)
slack_max.append(slack_max_prostate)
slack_min.append(slack_min_prostate)

c.append(c_blad)
slack_max.append(slack_max_blad)
slack_min.append(slack_min_blad)

c.append(c_rec)
slack_max.append(slack_max_rec)
slack_min.append(slack_min_rec)

c.append(c_naoesp)
slack_max.append(slack_max_naoesp)
slack_min.append(slack_min_naoesp)

del pontos_blad, pontos_blad_part, pontos_naoesp, pontos_prostate, pontos_rec, pontos_rec_part, pontos_ure
'''
tempo, custo, solution = resolve_reduzido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 10)
#plt.scatter(label_t,tempo, color='red', label = 'i = 10')
#plota_custo(custo)

tempo, custo, solution = resolve_reduzido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 100)
#plt.scatter(label_t,tempo, color='green', label = 'i = 100')
plota_custo(custo)

tempo, custo, solution = resolve_reduzido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 2466)
#plt.scatter(label_t,tempo, color='blue', label = 'i = 2466')
plota_custo(custo)
'''

tempo, custo, solution = build_BATD_exact(DmaxA, DminA, matriz_dose_full, MmaxA, MminA, t, c)
#tempo_ca, custo_ca, solution_ca = build_BATD_custo_alterado(DmaxA, DminA, matriz_dose_full, MmaxA, MminA, t, c, slack_max, slack_min)
#tempo, custo, solution = build_BATD_exact(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c)

'''
tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 0)
plt.scatter(label_t,tempo, color='red', label = 'Uretra')
del tempo, custo, solution

tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 1)
plt.scatter(label_t,tempo, color='green', label = 'Reto')
del tempo, custo, solution

tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 2)
plt.scatter(label_t,tempo, color='blue', label = 'Bexiga')
del tempo, custo, solution

tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 3)
plt.scatter(label_t,tempo, color='orange', label = 'Próstata')
del tempo, custo, solution

tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 4)
plt.scatter(label_t,tempo, color='black', label = 'Bexiga Parcial')
del tempo, custo, solution

tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 5)
plt.scatter(label_t,tempo, color='yellow', label = 'Reto Parcial')
del tempo, custo, solution

tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 6)
plt.scatter(label_t,tempo, color='brown', label = 'Tecido Saudavel')
'''
#tempo, custo, solution = resolve_tecido(Dmax, Dmin, matriz_dose_full, Mmax, Mmin, t, c, 0)


#plota_dose(matriz_dose_full,t)


#plota_custo(custo)

del c,c_ure,c_rec_part,c_blad_part,c_prostate,c_blad,c_rec,c_naoesp

print("Número de variáveis = ", mdl.number_of_variables)
if solution:
    print("Tem solução")
    print(mdl.solve_details)
    print(mdl.print_solution())
else:
    print("Não tem Solução")

for i in range(num_pencil_beams):
    tempo[i] = tempo[i].solution_value
    
   
for i in range(len(custo)):
    for j in range(len(custo[i])):
        custo[i][j] = custo[i][j].solution_value

plota_tempo(tempo)

finalTime = time.time()
print("Tempo de final = %d" %finalTime)
tempoTotal = finalTime - startTime