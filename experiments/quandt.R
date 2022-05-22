# Had to run R on in Colab.

my.qlr <- qlr.test(
    x ~ y, 
    data = list(), 
    from = 0, 
    to = length(x),
    sig.level = 0.05, 
    details = FALSE)
plot(my.qlr)