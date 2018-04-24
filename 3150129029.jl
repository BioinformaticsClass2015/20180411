
#####梯度法
function gradient_descent(f, g, x0; #a0=0.1,
        eg=0.001, ex=0.001, ef=0.001, maxIterations=128, verbose=true)#分号之后的参数为key word参数，需要键入命名才能使用
    x1 = x0
    d  = -g(x1...)
    f1 = f(x1...) #加三个点的形式不会报错？
    #inexact method for a
    a  = inexact_method(r, g, f1, -ng*ng, x1, d)
    x2 = x1 + a .* d#a是个标量，b是个向量
    f2 = f1
    ng = norm(d)
    nx = a*ng
    ni = 1
    while( (ng>eg || nx>ex || abds(f2-f1)>ef )&& ni<maxIterations)
        x1 = x2
        d  = -g(x1...)
        f1 = f2
        a  = a0#inecact_mathod()
        x2 = x1 + a .* d
        f2 = f(x2...)
        ng = norm(d) 
        nx = a*ng
        ni = ni+1
        if verbose
            println("Step ", ni, " x ", x2, " f ", f2)
            println("    d",d, " a ", a)
        end
    if ni >= maxIterations
        println("WARNING: interation exceed", maxIterations )
    else
        println("LOG: iteration converagea  after",ni  )
    end
    return x1, f2
    end    
end

function inexact_method(f, g, z0, zd0, xk, dk;t=0.5, e=0.5, z=2)
    a = 1
    da=dk*g((xk .+ a .* dx)...) #xk、dk都是向量
    while da<0
        a = z*a
        da=dk*g((xk .+ a .* dx)...) 
    end
    za=r((xk .+ a .* dx)...)
    while(za>z0+e*a*zd0)
        a = z*a
        za= r((xk .+ a .* dx)...)
    end
    return a
end

#####牛顿法
function newton_method(f, g, h, x0, e=0.001, maxIterration=128)
    x = x0
    xg= g(x...)
    xh= h(x...)
    i = 0
    if abs(norm(xg)) < e
        return x, f(x...)
    else
        while (abs(norm(xg)) > e && i < maxIterration)
            i =i + 1
            println(i, "\t", "x=", x, "\tg", xg, "\tH", xh)
            x = x + (-inv(xh)*xg)
            xg= g(x...)
            xh= h(x...)
        end
        return  x, f(x...)
    end
end
