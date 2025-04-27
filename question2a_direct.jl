using JuMP, Gurobi, DataFrames

# Create a new model
model = Model(Gurobi.Optimizer)

@variable(model, x1 >= 0);
@variable(model, x2 >= 0);
@variable(model, x3);

@constraint(model, x1 + x3 == 4);
@constraint(model, x2 - 2*x3 <= 2);

@objective(model, Min, 3*x1 + 5*x2 - x3);

optimize!(model)
value(x1)
value(x2)
value(x3)
JuMP.objective_value(model)

#create vector primal
model = Model(Gurobi.Optimizer)
@variable(model, x[1:4] >= 0);
b = [3, 5, -1, 1];
AT = [1 0 1 -1; -1 0 -1 1; 0 -1 2 -2];
c = [4, -4, -2];
@constraint(model, AT*x .>= c)
@objective(model, Min, b'*x)
optimize!(model)
objective_value(model);

# Create the dual
dual_model = Model(Gurobi.Optimizer)
@variable(dual_model, y[1:3] >= 0);
crs = @constraint(dual_model, AT'*y .<= b)
@objective(dual_model, Max, c'*y)
optimize!(dual_model)
objective_value(dual_model)
shadow_price.(crs)

# Create the dual
dual_model = Model(Gurobi.Optimizer)
@variable(dual_model, y1 >= 0);
@variable(dual_model, y2 >= 0)
@variable(dual_model, y3 >= 0)

@constraint(dual_model, y1 - y2 <= 3)
@constraint(dual_model, - y3 <= 5)
@constraint(dual_model, y1 - y2 + 2*y3 == -1)
# @constraint(dual_model, -y1 + y2 - 2*y3 <= -1)

@objective(dual_model, Max, 4*y1 - 4*y2 - 2*y3)

optimize!(dual_model)
objective_value(dual_model)


#alternative dual
dual_model = Model(Gurobi.Optimizer)
@variable(dual_model, y1);
@variable(dual_model, y2 <= 0);
@constraint(dual_model, y1 <= 3);
@constraint(dual_model, y2 <= 5);
@constraint(dual_model, y1 - 2*y2 == -1);
@objective(dual_model, Max, 4*y1 + 2*y2);
optimize!(dual_model)
objective_value(dual_model)
