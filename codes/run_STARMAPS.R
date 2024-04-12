age.list <- c('14', '17', '18', '19' ,'20', '22', '25')
save.file <- "lambda_1e-3/"
lambda <- 1e-3
library(Matrix)
library(glmnet)
library(lsei)
library(future.apply)
library(reticulate)
library(progressr)
library(futile.logger)
library(checkmate)
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}

logmsg <- function(..., level = "info") {
    msg = paste0(...)
    if (level == "debug") {
        futile.logger::flog.debug(msg)
    } else if (level == "info") {
        futile.logger::flog.info(msg)
    } else if (level == "warn") {
        futile.logger::flog.warn(msg)
    } else if (level == "error") {
        futile.logger::flog.error(msg)
    } else {
        stop("Log level not found!")
    }
}
log_to_file <- function(fn, logger = "ROOT", newfile = FALSE) {
    if (newfile) {
        if (file.exists(fn))
            # Delete file if it exists
        file.remove(fn)
    }
    tmp = futile.logger::flog.appender(futile.logger::appender.file(fn), name = logger)
}


log_to_file(paste0(save.file, 'my_log.log'), logger = "ROOT", newfile = FALSE)
logmsg("Start loading data.")

metadata.all <- read.csv('data/all_meta_data.csv', row.names=1)
dataset.all <- t(Matrix::readMM('data/data_matrix.mtx'))
rownames(dataset.all) <- row.names(metadata.all)

metadata.all <- read.csv('data/all_meta_data.csv', row.names=1)
dataset.all <- Matrix::readMM('data/data_matrix.mtx')

dataset.all <- t(dataset.all)

head(metadata.all)
dim(dataset.all)

rownames(dataset.all) <- row.names(metadata.all)
# construct X list
meta.unique <- unique(metadata.all[c('individual.x', 'area')])
batch.labels <- meta.unique$individual.x
region.labels <- meta.unique$area
X.list <- lapply(1:nrow(meta.unique), function(i){
  dataset.all[row.names(metadata.all[which((metadata.all$individual.x==batch.labels[i])&(metadata.all$area==region.labels[i])),]),]
})
batch.unique <- unique(batch.labels)
region.unique <- unique(region.labels)
logmsg(length(batch.unique), ' batches.')
logmsg(length(region.unique), ' regions.')
np <- import("numpy")
W <- np$load("data/init_W.npy")
H <- np$load("data/init_H.npy")
​
df <- data.frame(W)
row.names(df) <- row.names(metadata.all)
W.list <- lapply(1:nrow(meta.unique), function(i){
  as.matrix(df[row.names(X.list[[i]]),])
})
# parameters
k <- 40
​
# initialization
# objective function: 2078.88207121183
p <- ncol(X.list[[1]])                   
H.region.list <- lapply(region.unique, function(a) matrix(0, k, p))
names(H.region.list) <- region.unique
H.batch.list <- lapply(batch.unique, function(a) matrix(0, k, p))
names(H.batch.list) <- batch.unique
                       
logmsg("Finish loading data.")
objective_func <- function(X.list, W.list, H, H.batch.list, H.region.list, batch.labels, region.labels, lambda){
  nj.list = do.call(c, lapply(X.list, nrow))
  batch.unique <- unique(batch.labels)
  region.unique <- unique(region.labels)
  obj.main.list <- lapply(1:length(X.list), function(j){
    X.res <- X.list[[j]] - W.list[[j]] %*% (H + H.batch.list[[batch.labels[j]]] + H.region.list[[region.labels[j]]])
    norm(X.res, 'F')^2
  })
  obj.penalty.batch <- lapply(batch.unique, function(r){
    sum(abs(H.batch.list[[r]])) / nj.list[which(batch.labels == r)]
  })
  obj.penalty.region <- lapply(region.unique, function(r){
    sum(abs(H.region.list[[r]])) / nj.list[which(region.labels == r)]
  })
  obj.main <- sum(do.call(c, obj.main.list))
  obj.penalty <- lambda * (sum(do.call(c, obj.penalty.batch)) + sum(do.call(c, obj.penalty.region)))
  return((obj.main+obj.penalty) / sum(nj.list))
}
​
solve_H_region_list <- function(X.list, W.list, H, H.batch.list, batch.labels, region.labels, lambda){
  batch.unique <- unique(batch.labels)
  region.unique <- unique(region.labels)
  H.region.list <- lapply(region.unique, function(r){
      logmsg(r)
      ind.subset <- which(region.labels==r)
      m <- length(ind.subset)
      
      X.tmp <- lapply(1:m, function(j){
          ind.tmp <- ind.subset[j]
          X.list[[ind.tmp]] - W.list[[ind.tmp]] %*% (H + H.batch.list[[batch.labels[ind.tmp]]])
      })
      
      X.concat <- as.matrix(do.call(rbind, X.tmp))
      W.concat <- as.matrix(do.call(rbind, W.list[ind.subset]))
      return(solve_H_constraint(X.concat, W.concat, as.matrix(H), lambda))
  })
  names(H.region.list) <- region.unique
  return(H.region.list)
}
​
solve_H_batch_list <- function(X.list, W.list, H, H.region.list, batch.labels, region.labels, lambda){
  batch.unique <- unique(batch.labels)
  region.unique <- unique(region.labels)
  H.batch.list <- lapply(batch.unique, function(r){
      logmsg(r)
      ind.subset <- which(batch.labels==r)
      m <- length(ind.subset)
      X.tmp <- lapply(1:m, function(j){
          ind.tmp <- ind.subset[j]
          return(X.list[[ind.tmp]] - W.list[[ind.tmp]] %*% (H + H.region.list[[region.labels[ind.tmp]]]))
      })
      X.concat <- as.matrix(do.call(rbind, X.tmp))
      W.concat <- as.matrix(do.call(rbind, W.list[ind.subset]))
      return(solve_H_constraint(X.concat, W.concat, as.matrix(H), lambda))
  })
  names(H.batch.list) <- batch.unique
  return(H.batch.list)
}
​
solve_H_constraint <- function(X, W, H, lambda){
    p <- ncol(X)
    H.constraint <- do.call(cbind, future.apply::future_lapply(1:p, function(l){
        beta <- try(glmnet(W, X[, l], lambda = lambda, alpha=1, family = "gaussian",
        lower.limits=-1/2*H[, l], intercept = FALSE, standardize = FALSE)$beta) # k times p
        if(inherits(beta, "try-error")){
            beta <- as.numeric(rep(0, nrow(H)))
            }
        return(beta)
    }))
  return(H.constraint)
}
                       
solve_H <- function(X.list, H, W.list, H.batch.list, H.region.list, batch.labels, region.labels, lambda){
    p = ncol(X.list[[1]])
    m = length(W.list)
    nj.list = do.call(c, lapply(X.list, nrow))  # avoid repeated calculation
​
    X.list.tmp <- lapply(1:m, function(j) X.list[[j]] - W.list[[j]] %*% (H.batch.list[[batch.labels[j]]] +
                                                                         H.region.list[[region.labels[j]]]))
    logmsg('Start update of H.')
    H.update = do.call(rbind, future.apply::future_lapply(1:p, function(l) {
        A = do.call(rbind, W.list)
        B = do.call(c, lapply(1:m, function(j) {
          X.list.tmp[[j]][, l]  # nj*1
        }))
        x <- try(lsei::nnls(a = as.matrix(A), b = B)$x)
        # res <- try(expression_to_get_data)
        if(inherits(x, "try-error")){
            x <- as.numeric(H[, l])
        }
        return(x)
    }))
  checkmate::assert_true(any(is.na(H.update)) == F)
  
  return(t(H.update))
}
                       
solve_W <- function(X, W, H.cur) {
    # check size X n*p, H.cur p*r, lambda p, b p
    W.new = do.call(rbind, future.apply::future_lapply(1:nrow(X), function(i) {
        x <- try(lsei::pnnls(a = as.matrix(H.cur), b = X[i, ], sum = 1)$x)
        if(inherits(x, "try-error")){
            x <- as.numeric(W[i, ])
        }
        return(x)
    }))
    # checkmate::assert_true(any(is.na(W)) == F)
    return(W.new)
}
​
​
solve_W_list <- function(X.list, H, W.list, H.batch.list, H.region.list, batch.labels, region.labels){
    m = length(X.list)
    W.list <- lapply(1:m, function(j){
        H.cur <- t(H + H.batch.list[[batch.labels[j]]] + H.region.list[[region.labels[j]]])
        return(solve_W(X.list[[j]], W.list[[j]], H.cur))
    })
    return(W.list)
}                    
                       
solve_subproblem <- function(params.to.update = c("W", "H.region", "H", "H.batch"), X.list,
    W.list, H, H.batch.list, H.region.list, batch.labels, region.labels, lambda, verbose = T) {
    params.to.update = match.arg(params.to.update)
    m = length(X.list)
​
    if (params.to.update == "W") {
        W.list <- solve_W_list(X.list, H, W.list, H.batch.list, H.region.list, batch.labels, region.labels)
    } else if (params.to.update == "H") {
        H <- solve_H(X.list, H, W.list, H.batch.list, H.region.list, batch.labels, region.labels, lambda)
    } else if (params.to.update == "H.region") {
        H.region.list <- solve_H_region_list(X.list, W.list, H, H.batch.list, batch.labels, region.labels, lambda)
    } else {
        H.batch.list <- solve_H_batch_list(X.list, W.list, H, H.region.list, batch.labels, region.labels, lambda)
    }
    return(list(W.list = W.list, H.batch.list = H.batch.list, H.region.list = H.region.list, H = H))
}
                       
params.list <- list(W.list = W.list, H.batch.list = H.batch.list, H.region.list = H.region.list, H = H)
# params.list <- readRDS(paste0(save.file, "iter1.rds"))
plan(multicore, workers=parallel::detectCores() - 1)
options(future.globals.maxSize=5000*1024^2)
set.seed(1)
obj.history <- objective_func(X.list, params.list$W.list, params.list$H, params.list$H.batch.list, 
                              params.list$H.region.list, batch.labels, region.labels, lambda)
obj <- obj.history
logmsg("Initial objective function ", obj)
tol <- 5e-05 #~0.1
for (iter in 1:50){
    obj.old = obj
    for (params.to.update in c("H.batch", "H.region", "H", "W")){
        logmsg(params.to.update)
        params.list = solve_subproblem(params.to.update = params.to.update,
            X.list = X.list, W.list = params.list$W.list, H.batch.list = params.list$H.batch.list,
            H.region.list = params.list$H.region.list, H = params.list$H,
            batch.labels = batch.labels, region.labels=region.labels, lambda=lambda)
        obj <- objective_func(X.list, params.list$W.list, params.list$H, params.list$H.batch.list,
                       params.list$H.region.list, batch.labels, region.labels, lambda)
        logmsg(obj)
        }
    obj.history = c(obj.history, obj)
    delta = abs(obj - obj.old)/mean(c(obj, obj.old))
    if (delta < tol) {
        logmsg("Converge at iter ", iter, ", obj delta = ", delta)
        converge = T
        break
    }
    saveRDS(params.list, file=paste0(save.file, "iter", iter, ".rds"))
    logmsg("Save the whole iteration.")
    saveRDS(obj.history, file=paste0(save.file, "obj_history.rds"))
}
H[1:5, 1:5]
                         
