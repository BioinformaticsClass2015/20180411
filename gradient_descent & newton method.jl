
#####回溯法搜索步长
function inexact_method(f, g, ϕ0, ϕd0, xk, dk ;τ=0.5, ϵ=0.5, ζ=2 )
    α = 1
   dα = dk' * g((xk .+ α .* dk)...)
    while dα<0
        α = ζ*α
        dα= dk' * g((xk .+ α .*dk)...)
    end
    ϕα= f((xk .+ α .* dk)...)
    while (ϕα> ϕ0 + ϵ * ϕd0 *α)
         α= τ*α
        ϕα= f((xk .+ α .* dk)...)
    end
    return α    
end

######最速下降法搜索目标函数的极小点和极小值
function gradient_descent(f,g,x0; ϵg=0.0001,ϵx=0.0001,ϵf=0.0001,maxIteration=1300,verbose=true)
    x1 = x0
    d  = -g(x1...)
    f1 = f(x1...)
    g1 = g(x1...)
    ###inexact_method for α
    α = inexact_method(f, g, f1, -g1'*g1,x1,d)  # 回溯法搜索步长
    ng = norm(d)
    nx = α*ng
    ni = 1
    x2 = x1+α .* d 
    f2 = f(x2...)
    g2 = g(x2...)
    println("Step",ni,"  x=",x2,"  f=",f2,"  g=",g2)
    println("        d=",d, " α=",α )
    while (ng>ϵg || nx>ϵx || abs(f2-f1)>ϵf )&& ni<maxIteration   
        x1 = x2
         d = -g(x1...)
        f1 = f2
        g1=g(x1...)
        α = inexact_method(f, g, f1, -g1'*g1,x1,d) # 回溯法搜索步长
        x2 = x1+α .* d 
        f2 = f(x2...)
        g2=g(x2...)
        ng = norm(d)
        nx = α*ng
        ni += 1 
        if verbose
            println("step ", ni, "  x= ",x2, "  f ", f2,"  g=",g2)
            println("        d=",d, " α=",α )
        end
        if ni == maxIteration
        println("WARNING：iteration exceeds ",maxIteration, "step convergence before")
       end
    end
    return x2,f2
end

###最速下降法实现Rosenbrock函数极小点，初始点（x,y）=[-2,2], 计算精确度为0.0001
gradient_descent(
    (x,y)->(1-x)^2+100*(y-x^2)^2,
    (x,y)->[-2*(1-x)-400*(y-x^2)*x, 200*(y-x^2)],
    [-2,2]
)

####牛顿法搜索目标函数的极小点和极小值
function newton(f,g,H,x0;ϵ=0.0001,maxIterration=128)
    x = x0
    ng=norm(g(x...))
    ni=1
    println("step = ",ni,"\t","x = ",x,"\t","gx= ",g(x...), "\t","f=",f(x...))
    println("\t","\t","H=",H(x...), "\t", "H- =",inv(H(x...)),"\t","H-g=",inv(H(x...)) *g(x...))
    while abs(ng) > ϵ && ni<maxIterration
        if det(H(x...))==0        ##判断黑塞矩阵是否可逆(行列式为0不可逆)；
            println("ERROR : H Matrix irreversible!","\r","Can't use Newton! ")
        else 
            x = x - inv(H(x...)) *g(x...)
            ng=norm(g(x...))
            ni += 1
            println("step = ",ni,"\t","x = ",x,"\t","gx= ",g(x...), "\t","f=",f(x...))
            println("\t","\t","H=",H(x...), "\t", "H- =",inv(H(x...)),"\t","H-g=",inv(H(x...)) *g(x...))
        end
    end
    return x,f(x...)
end


###牛顿法实现Rosenbrock函数极小点，初始点（x,y）=[-2,2], 计算精确度为0.0001
newton(
    (x,y)->(1-x)^2+100*(y-x^2)^2,
    (x,y)->[-2*(1-x)-400*(y-x^2)*x, 200*(y-x^2)],
    (x,y)->[2-400*(y-3*x^2) -400*x; -400*x 200],
    [-2,2]
)

#####回溯法搜索步长
function inexact_method(f, g, H, ϕ0, ϕd0, xk, dk ;τ=0.5, ϵ=0.5, ζ=2 )
    α = 1
   dα =  g((xk .+ α .* dk)...)*dk'
    while dα<0
        α = ζ*α
        dα=g((xk .+ α .*dk)...)*dk'
    end
    ϕα= f((xk .+ α .*dk)...)
    while (ϕα> ϕ0 + ϵ * ϕd0 *α)
         α= τ*α
        ϕα= f((xk .+ α .* dk)...)
    end
    return α    
end

####Levenberg-Marquardt 改进算法
function Levenberg_Marquardt(f,g,H,x0;u=eps(),ϵ=0.0001,maxIterration=1500)
    I=[1 0;0 1]
    x = x0
    ng=norm(g(x...))
    ni=1
    println("step = ",ni,"\t","x = ",x,"\t","gx= ",g(x...), "\t","f=",f(x...))
    println("\t","\t","H=",H(x...), "\t", "H- =",inv(H(x...)),"\t","H-g=",inv(H(x...)) *g(x...))
    while abs(ng) > ϵ && ni<maxIterration
        if det(H(x...))==0        ##判断黑塞矩阵是否可逆(行列式为0不可逆)；
            println("ERROR : H Matrix irreversible!","\r","Can't use Newton! ")
        else 
            fx=f(x...)
            gx=g(x...)
            d =-inv(H(x...)+u*I) *g(x...)
            ϕd=-gx'*inv(H(x...)+u*I)*gx
            α= 0.01 #inexact_method(f, g, H, fx, ϕd, x, d)
            x = x - α*inv(H(x...)+u*I) *g(x...)
            ng=norm(g(x...))
            ni += 1
            println("step = ",ni,"\t","x = ",x,"\t","gx= ",g(x...), "\t","f=",f(x...))
            println("\t","\t","H=",H(x...), "\t", "H- =",inv(H(x...)),"\t","H-g=",inv(H(x...)) *g(x...))
        end
    end
    return x,f(x...)
end

###Levenberg-Marquardt 改进算法实现Rosenbrock函数极小点，初始点（x,y）=[-2,2], 计算精确度为0.0001
Levenberg_Marquardt(
    (x,y)->(1-x)^2+100*(y-x^2)^2,
    (x,y)->[-2*(1-x)-400*(y-x^2)*x, 200*(y-x^2)],
    (x,y)->[2-400*(y-3*x^2) -400*x; -400*x 200],
    [-2,2]
)
