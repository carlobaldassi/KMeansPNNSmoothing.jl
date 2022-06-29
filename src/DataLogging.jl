module DataLogging

export
    init_logger,
    get_logger,
    pop_logger!

mutable struct DataLogger
    closed::Bool
    io::IO
    prefix_stack::Vector{String}
    DataLogger(filename::AbstractString, mode::String = "a") = new(false, open(filename, mode), String[])
end

function Base.close(dlogger::DataLogger)
    @assert !dlogger.closed
    close(dlogger.io)
    dlogger.closed = true
    return
end

const logging_on = parse(Bool, get(ENV, "DATALOGGING", "false"))
if logging_on && Threads.nthreads() > 1
    @warn "DataLogging does not work well with multithreading"
end

let logger_stack = DataLogger[]
    global function init_logger(filename::AbstractString, mode::String = "a")
        logger = DataLogger(filename, mode)
        push!(logger_stack, logger)
        return logger
    end

    global function get_logger()
        n = length(logger_stack)
        return n == 0 ? nothing : logger_stack[n]
    end

    global function pop_logger!()
        length(logger_stack) == 0 && error("logger stack is empty, cannot pop")
        close(logger_stack[end])
        return pop!(logger_stack)
    end
end

macro push_prefix!(prefix)
    logging_on || return Expr(:block)

    pexpr = :(push!(dl.prefix_stack, string(begin local value = $(esc(prefix)) end)))
    return quote
        dl = get_logger()
        if dl ≢ nothing
            @assert !dl.closed
            prefix = join(dl.prefix_stack, " ")
            $pexpr
        end
    end
end

macro pop_prefix!()
    logging_on || return Expr(:block)
    return quote
        dl = get_logger()
        if dl ≢  nothing
            @assert !dl.closed
            pop!(dl.prefix_stack)
        end
    end
end

macro log(args...)
    logging_on || return Expr(:block)

    blk = Expr(:block)
    for arg in args
        push!(blk.args, :(print(dl.io, ' ', string(begin local value = $(esc(arg)) end))))
    end
    return quote
        dl = get_logger()
        if dl ≢ nothing
            @assert !dl.closed
            prefix = join(dl.prefix_stack, " ")
            print(dl.io, prefix, " >")
            $blk
            println(dl.io)
        end
    end
end

macro exec(args...)
    if logging_on
        return Expr(:block, map(esc, args)...)
    else
        return Expr(:block)
    end
end

end # module
