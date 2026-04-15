#include <casadi/casadi.hpp>
#include <iostream>

using namespace casadi;

int main() {
    // 1. Définir la fonction
    SX x = SX::sym("x", 4);
    SX u = SX::sym("u", 2);

    double dt = 0.1;
    SX A = SX::zeros(4,4);
    A(0,0) = 1.0; A(0,2) = dt;
    A(1,1) = 1.0; A(1,3) = dt;
    A(2,2) = 1.0;
    A(3,3) = 1.0;

    SX B = SX::zeros(4,2);
    B(2,0) = dt;
    B(3,1) = dt;

    SX x_next = mtimes(A, x) + mtimes(B, u);

    Function dyn_fun = Function("dyn_fun", {x, u}, {x_next});

    // 2. Utiliser CodeGenerator pour sortir un .c
    CodeGenerator cg("model_dynamics");
    cg.add(dyn_fun);
    cg.generate();

    std::cout << "Code généré !" << std::endl;

    return 0;
}
