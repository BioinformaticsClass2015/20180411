y<-function(x1,x2,x3,x4)
  {(x1+10*x2)^2+5*(x3-x4)^2+(x2-2*x3)^4+10*(x1-x4)^4}#原函数



g<-function(x1,x2,x3,x4)
{matrix(c(2*(x1+10*x2)+40*(x1-x4)^3,20*(x1+10*x2)+4*(x2-2*x3)^3,
          10*(x3-x4)-8*(x2-2*x3)^3,-10*(x3-x4)-40*(x1-x4)^3),4,1)}#梯度


f<-function(x1,x2,x3,x4)
{matrix(c(2+120*(x1-x4)^2,20,0,-120*(x1-x4)^2,20,200+12*(x2-2*x3)^2,
          -24*(x2-2*x3)^2,0,0,-24*(x2-2*x3)^2,10+48*(x2-2*x3)^2,-10,
          -120*(x1-x4)^2,0,-10,10+120*(x1-x4)^2),4,4,byrow=TRUE)}#黑塞矩阵


niudun<-function(x1,x2,x3,x4)
{m<-matrix(c(x1,x2,x3,x4),4,1)
n<-m-solve(f(m[1],m[2],m[3],m[4]))%*%g(m[1],m[2],m[3],m[4])
while(y(n[1],n[2],n[3],n[4])<y(m[1],m[2],m[3],m[4]))
{m<-n
n<-m-solve(f(m[1],m[2],m[3],m[4]))%*%g(m[1],m[2],m[3],m[4])}
print(n)
print(y(m[1],m[2],m[3],m[4]))
}

niudun<-function(x1,x2,x3,x4)
{m<-matrix(c(x1,x2,x3,x4),4,1)
n<-m-ginv(f(m[1],m[2],m[3],m[4]))%*%g(m[1],m[2],m[3],m[4])
while(y(n[1],n[2],n[3],n[4])<y(m[1],m[2],m[3],m[4]))
{m<-n
n<-m-ginv(f(m[1],m[2],m[3],m[4]))%*%g(m[1],m[2],m[3],m[4])}
print("极值点")
print(m)
print(paste("函数值",y(m[1],m[2],m[3],m[4])))
}#主函数



