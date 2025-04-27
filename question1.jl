using JuMP, Gurobi, DataFrames

# Create a new model
model = Model(Gurobi.Optimizer)

@variable(model, xA[1:4] >= 0.0)
@variable(model, xB[1:4] >= 0.0)
@variable(model, xC[1:4] >= 0.0)

costs = [17, 20, 24]
capacities = [800, 600, 700]
sc = [3 2 5 7; 6 4 8 3; 9 1 5 4]
demands = [300, 500, 400, 600];

#capacity constraints
capA = @constraint(model, sum(xA) <= capacities[1])
capB = @constraint(model, sum(xB) <= capacities[2])
capC = @constraint(model, sum(xC) <= capacities[3])
#demand constraints
for i in 1:4
    @constraint(model, xA[i] + xB[i] + xC[i] >= demands[i])
end

#objective function
@objective(model, Min, 
    xA'*(sc[1,:] .+ costs[1]) + xB'*(sc[2,:] .+ costs[2]) + xC'*(sc[3,:] .+ costs[3]))
    
optimize!(model)
JuMP.objective_value(model)
# #alternative optima
# @constraint(model, xA[2] == 500)
# optimize!(model)
# JuMP.objective_value(model)

#make a matrix in your output
result = [value.(xA)'; value.(xB)'; value.(xC)']
result
#print matrix with column and row names
df = DataFrame(result, [:Onyx, :Treble, :Hilton, :Dean])
insertcols!(df, 1, :City => [:Aberdeen, :Birmingham, :Cardiff])

#check the sensitivity 
println("Shadow prices for the capacity constraints")
println("A: ", JuMP.shadow_price(capA))
println("B: ", JuMP.shadow_price(capB))
println("C: ", JuMP.shadow_price(capC))
JuMP.objective_value(model)

#so let's change capC to 800 and see what happens
set_normalized_rhs(capC, capacities[3] + 100)
optimize!(model)
JuMP.objective_value(model)
#but change B by one
set_normalized_rhs(capB, capacities[2] + 1)
optimize!(model)
JuMP.objective_value(model) #cost got better by 1
#change B by 100
set_normalized_rhs(capB, capacities[2] + 100)
optimize!(model)
JuMP.objective_value(model) #cost got better by 100 #also that
#change B by 800
set_normalized_rhs(capB, capacities[2] + 800)
optimize!(model)
JuMP.objective_value(model) #cost got better by 200
println("B: ", JuMP.shadow_price(capB))
