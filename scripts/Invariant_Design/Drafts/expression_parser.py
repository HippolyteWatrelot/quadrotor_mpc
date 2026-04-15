from casadi import cos, sin, tan, MX
from sympy import *
import numpy as np

operators = ['Add', 'Mul', 'Pow', 'cos', 'sin', 'tan', 'Symbol']
tree = {"op": None, "subexps": dict(), 'expression': None}


def get_node(chain, tree):
    if len(chain) == 0:
        return tree
    else:
        subtree = tree
        for elt in chain:
            subtree = subtree['subops'][elt]
        return subtree
        
        
def base(exp):
    
        
        
def get_parenthesis(expression):
    indexes = []
    i = 0
    while i < len(expression):
        if expression[i] == '(':
            counter = 1
            j = i+1
            while True:
                if expression[j] == '(':
                    counter += 1
                elif expression[j] == ')':
                    counter -=1
                if counter == 0:
                    indexes.append((i, j))
                    break
                j += 1
        i = j+1
    return indexes


def get_subs(expression):
    indexes = get_parenthesis(expression)
    exp_array = expression.split('(')
    op = exp_array[0]
    subexp_array = content[len(op)+1:-1]
    
    return subops, subexpressions


def update_tree(expression, chain=[]):
    '''init at empty chain and full expression'''
    global tree
    subtree = tree.copy()
    if len(chain) == 0:
        content = srepr(expression)
        exp_array = content.split('(')
        chain = [exp_array[0] + str(0)]
    rev_chain = chain.copy()
    rev_chain.reverse()
    for elt in chain:
        subtree = subtree['subexps'][elt]        # until chain bottom
    subops, subexpressions = get_subs(expression)    # subops is dict of dicts
    subtree.update({"op": chain[-1], "subexps": subexpressions, "expression": expression})
    midtree = subtree.copy()
    for elt in rev_chain:
        chain.remove(elt) 
        midtree = get_node(chain, tree)
        midtree["subops"][elt].update(subtree)
        subtree = midtree
    tree = subtree
    if len(subexpressions) == 1 and base(subexpression[0]):
        '''Analyze base expression'''
        return
    else:
        for (so, se) in (subops, subexpressions):
            new_chain = chain + so
            update_tree(se, new_chain)


def sub_formulas(array):
    


def sympexp2caTree(exp):
    global operators
    content = srepr(exp)     # To be used outside
    
    Add(Mul(Add(Symbol('g'), Mul(Add(Mul(Integer(-1), Symbol('d_z'), Symbol('gamma_z')), Mul(Symbol('gamma_z'), Symbol('uz'))), Pow(cos(Symbol('pitch')), Integer(-1)), Pow(cos(Symbol('roll')), Integer(-1)))), Add(Mul(sin(Symbol('pitch')), sin(Symbol('yaw')), cos(Symbol('roll'))), Mul(Integer(-1), sin(Symbol('roll')), cos(Symbol('yaw')))), sin(Symbol('yaw'))), Mul(Add(Symbol('g'), Mul(Add(Mul(Integer(-1), Symbol('d_z'), Symbol('gamma_z')), Mul(Symbol('gamma_z'), Symbol('uz'))), Pow(cos(Symbol('pitch')), Integer(-1)), Pow(cos(Symbol('roll')), Integer(-1)))), Add(Mul(sin(Symbol('pitch')), cos(Symbol('roll')), cos(Symbol('yaw'))), Mul(sin(Symbol('roll')), sin(Symbol('yaw')))), cos(Symbol('yaw'))))
    
    exp_array = content.split('(')
    op = exp_array[0]
    tree["op"] = op
    subexp_array = content[len(op)+1:-1]
    nodes = sub_formulas(subexp_array)
    if op == 'Symbol' and len(nodes) == 1 and nodes[0] in ["'", '"']:
        
    else:
        for node in nodes:
            tree["objects"][node]
            
            
def sympexp2caFormula(s_exp):
    
