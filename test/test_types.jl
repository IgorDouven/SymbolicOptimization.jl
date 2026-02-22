@testset "Core Types" begin
    
    @testset "Constant" begin
        c1 = Constant(3.14)
        @test c1.value == 3.14
        @test isconstant(c1)
        @test !isvariable(c1)
        @test !isfunction(c1)
        @test isterminal(c1)
        @test arity(c1) == 0
        @test isempty(children(c1))
        
        # Construction from Int
        c2 = Constant(42)
        @test c2.value == 42.0
        @test c2.value isa Float64
    end
    
    @testset "Variable (untyped)" begin
        v = Variable(:x)
        @test v.name == :x
        @test vartype(v) === nothing
        @test !istyped(v)
        @test isvariable(v)
        @test !isconstant(v)
        @test !isfunction(v)
        @test isterminal(v)
        @test arity(v) == 0
        @test isempty(children(v))
    end
    
    @testset "Variable (typed)" begin
        v = Variable(:ps, :Vector)
        @test v.name == :ps
        @test vartype(v) == :Vector
        @test istyped(v)
        @test isvariable(v)
        
        # Different types
        v2 = Variable(:n, :Scalar)
        @test vartype(v2) == :Scalar
    end
    
    @testset "FunctionNode" begin
        x = Variable(:x)
        c = Constant(1.0)
        
        # Binary function
        add = FunctionNode(:+, x, c)
        @test add.func == :+
        @test length(add.children) == 2
        @test isfunction(add)
        @test !isconstant(add)
        @test !isvariable(add)
        @test !isterminal(add)
        @test arity(add) == 2
        @test children(add) == [x, c]
        
        # Unary function
        sin_x = FunctionNode(:sin, x)
        @test arity(sin_x) == 1
        
        # Construction with vector
        mul = FunctionNode(:*, [x, c])
        @test mul.func == :*
        @test arity(mul) == 2
        
        # Nested construction
        nested = FunctionNode(:sin, FunctionNode(:+, x, c))
        @test arity(nested) == 1
        @test isfunction(nested.children[1])
    end
    
    @testset "Equality" begin
        # Constants
        @test Constant(1.0) == Constant(1.0)
        @test Constant(1.0) != Constant(2.0)
        
        # Variables
        @test Variable(:x) == Variable(:x)
        @test Variable(:x) != Variable(:y)
        @test Variable(:x, :Scalar) == Variable(:x, :Scalar)
        @test Variable(:x, :Scalar) != Variable(:x, :Vector)
        @test Variable(:x) != Variable(:x, :Scalar)  # typed vs untyped
        
        # FunctionNodes
        x = Variable(:x)
        c = Constant(1.0)
        @test FunctionNode(:+, x, c) == FunctionNode(:+, Variable(:x), Constant(1.0))
        @test FunctionNode(:+, x, c) != FunctionNode(:-, x, c)
        @test FunctionNode(:+, x, c) != FunctionNode(:+, c, x)  # order matters
        
        # Different types never equal
        @test Constant(1.0) != Variable(:x)
        @test Variable(:x) != FunctionNode(:f, Variable(:x))
    end
    
    @testset "Hashing" begin
        # For use in Sets and Dict keys
        nodes = Set{AbstractNode}()
        push!(nodes, Constant(1.0))
        push!(nodes, Constant(1.0))  # duplicate
        push!(nodes, Variable(:x))
        push!(nodes, Variable(:x))  # duplicate
        
        @test length(nodes) == 2
        @test Constant(1.0) in nodes
        @test Variable(:x) in nodes
        
        # Dict usage
        d = Dict{AbstractNode, Int}()
        d[Constant(1.0)] = 1
        d[Variable(:x)] = 2
        @test d[Constant(1.0)] == 1
        @test d[Variable(:x)] == 2
    end
end
