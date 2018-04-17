#梯度下降法
function inexact_method( f ,g ,ϕ0 ,ϕd0 ,xk , dk ;τ=0.5, ϵ=0.5, ζ=2)##回溯搜索法
    a=1
    da=dk' * g((xk .+ a .* dk)...)    ###  对f(xk .+ a .* dk)求导
    while ( da<0)
        a=ζ*a
        da=dk'* g((xk .+a .* dk)...)
    end
    ϕa=f((xk .+a .* dk)...)
    while ( ϕa > ϕ0 + ϵ * a * ϕd0 )
        a=τ*a
        ϕa=f((xk .+a .* dk)...)
    end
    return a 
end

function gradient_descent(f,g,x0;
        eg=0.001,ex=0.001,ef=0.001,maxIterations=128,verbose=true)
    x1 = x0
    d  = -g(x1...)
    f1 = f(x1...)
    g1 = g(x1...)
    println("Step " ,"  x=",x1,"  f=",f1,"  g=",g1)
    ###Inexact method for a
    a  = inexact_method(f,g,f1,-g1'*g1,x1,d)
    ng = norm(d)   ##  梯度的范数<ϵ。
    nx = a*ng      ##  a*d的范数<ϵ。
    ni = 1
    x2 = x1+a.*d
    f2 = f(x2...)
    g2 = g(x2...)
    println("Step",ni,"  x=",x2,"  f=",f2,"  g=",g2,"  a=",a)
    while (ng>eg || nx>ex || abs(f2-f1)>ef)&& ni < maxIterations#!(ng<eg && nx<ex && abs(f2-f1)<ef) && ni < maxIterations
        x1 = x2
        d  = -g(x1...)
        f1 = f2
        g1 = g(x1...)
        a  = inexact_method(f,g,f1,-g1'*g1,x1,d)
        ng = norm(d)
        nx = a*ng
        ni += 1
        x2 = x1+a .*d
        f2 = f(x2...)
        g2  = g(x2...)
        if verbose
            println("Step",ni,"  x=",x2,"  f=",f2,"  g=",g2)
            println("        d=",d, " a=",a )
        end
        if ni ==maxIterations
            println("WARNING: Interations exceeds ",maxIterations," steps")
        #else 
            #println(" LOG:  Iterations converges after ",ni," steps")
        end
    end
    return(x2,f2,g2)
end
#牛顿法
function Newton(f,g,h,x0,ϵ=eps(),maxIterations=128)
    x=x0
    gx=g(x...)
    i=0
    if  norm(gx)<eps()
        return x,f(x...),g(x...)
    else
        while norm(gx)>ϵ&&i<=maxIterations
            i=i+1
            hx=h(x...)
            inv_hx=inv(hx)
            println("Step:$i x=$x \t f(x)=$(f(x...)) \t g(x)=$(g(x...))")
            x=x-inv_hx * gx
            gx=g(x...)
            if i >maxIterations
                println("WARNING: Interations exceeds ",maxIterations," steps")
            end
        end
        return x,f(x...),g(x...),i
    end
end

#  Levenberg-Marquardt（修正）
function Inexact_Method( f ,g ,inv_h,ϕ0 ,ϕd0 ,xk , dk ;τ=0.5, ϵ=0.5, ζ=2)##回溯搜索法
    a=1
    da=dk' * g((xk + a .* inv_h * dk)...)    ###  对f(xk .+ a .* dk)求导
    while ( da<0)
        a=ζ*a
        da=dk'* g((xk + a .* inv_h * dk)...)
    end
    ϕa=f((xk + a .* inv_h * dk)...)
    while ( ϕa > ϕ0 + ϵ * a * ϕd0 )
        a=τ*a
        ϕa=f((xk + a .* inv_h * dk)...)
    end
    return a 
end

function Levenberg_Marquardt(f,g,h,x0,ϵ=eps(),maxIterations=128;eg=0.0001,ex=0.0001,ef=0.001,verbose=true)
    x1 = x0
    d  = -g(x1...)
    f1 = f(x1...)
    g1 = g(x1...)
    h1 = h(x1...)
    eigen = eig(h1)[1]
    λ=maximum(eigen)
    while  !(eigen[1] > 0&&eigen[2] > 0)
        H1    = h1+diagm([λ,λ])
        eigen = eig(H1)[1]
        h1    = H1
        λ=maximum(eigen)
    end
    inv_h1=inv(h1)
    println("Step " ,"  x=",x1,"  f=",f1,"  g=",g1)
    if  norm(g1)<eps()
        return x1,f(x1...),g(x1...)
    else
        ###Inexact method for a
        a  = Inexact_Method(f,g,inv_h1,f1,-g1'*inv_h1*g1,x1,d)
        ng = norm(d)   ##  梯度的范数<ϵ。
        nx = a*ng      ##  a*d的范数<ϵ。
        ni = 1
        x2 = x1 - a .* inv_h1 * g1
        f2 = f(x2...)
        g2 = g(x2...)
        println("Step",ni,"  x=",x2,"  f=",f2,"  g=",g2,"  a=",a)
        while (ng>eg || nx>ex || abs(f2-f1)>ef)&& ni < maxIterations#!(ng<eg && nx<ex && abs(f2-f1)<ef) && ni < maxIterations
            x1 = x2
            d  = -g(x1...)
            f1 = f2
            g1 = g(x1...)
            h1 = h(x1...)
            eigen = eig(h1)[1]
            λ=maximum(eigen)
            while  sum(eigen.>0)!=2
                H1    = h1+diagm([λ,λ])
                eigen = eig(H1)[1]
                h1    = H1
                λ=maximum(eigen)
            end
            inv_h1=inv(h1)
            a  = Inexact_Method(f,g,inv_h1,f1,-g1'*inv_h1*g1,x1,d)
            ng = norm(d)
            nx = a*ng
            ni += 1
            x2 = x1 - a .* inv_h1 * g1
            f2 = f(x2...)
            g2 = g(x2...)
            if verbose
                println("Step",ni,"  x=",x2,"  f=",f2,"  g=",g2)
                println("        d=",d, " a=",a )
            end
            if ni ==maxIterations
                println("WARNING: Interations exceeds ",maxIterations," steps")
            end
        end
    end
    return(x2,f2,g2)
end

##非正定的矩阵
Levenberg_Marquardt(
    (x,y)-> x^4+x*y+(1+y)^2,
    (x,y)->[4*x^3+y , x+2*(1+y)],
    (x,y)->[12*x^2  1 ; 1  2],
    [0,0]
)

##非正定的矩阵
gradient_descent(
    (x,y)-> x^4+x*y+(1+y)^2,
    (x,y)->[4*x^3+y , x+2*(1+y)],
    [0,0]
)

##非正定的矩阵
Newton(
    (x,y)-> x^4+x*y+(1+y)^2,
    (x,y)->[4*x^3+y , x+2*(1+y)],
    (x,y)->[12*x^2  1 ; 1  2],
    [0,0]
)
