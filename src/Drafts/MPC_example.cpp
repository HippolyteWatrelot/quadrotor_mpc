#include <casadi/casadi.hpp>

using namespace casadi;

int main() {
    SX x = SX::sym("x");
    SX f_expr = pow(x, 2);
    SX g_expr = x;

    // Options du solver
    Dict opts;
    opts["ipopt_options"] = {{"linear_solver", "mumps"}};  // ✅ C’est ici !
    opts["ipopt_options"]["print_level"] = 0;               // Option pour réduire l’output

    // Création du solveur
    Function nlp = nlpsol("solver", "ipopt",
                          {{"x", x}, {"f", f_expr}, {"g", g_expr}},
                          opts);

    // Arguments du NLP
    std::map<std::string, DM> arg, res;
    arg["x0"] = DM(0);
    arg["lbx"] = DM(-1);
    arg["ubx"] = DM(1);
    arg["lbg"] = DM(0);
    arg["ubg"] = DM(0);

    // Résolution
    res = nlp(arg);
    std::cout << "Solution: " << res["x"] << std::endl;

    return 0;
}
