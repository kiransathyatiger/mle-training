import ingestdata
import logger
import score
import train

if __name__ == "__main__":
    args = ingestdata.parse_args()
    logger = logger.configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )
    print("logging done")

    ingestdata.run(args, logger)
    print("ingestion done")

    args = train.parse_args()
    train.run(args, logger)

    args = score.parse_args()
    # logger = logger.configure_logger(log_level=args.log_level, log_file=args.log_path, console=not args.no_console_log,)
    score.run(args, logger)
