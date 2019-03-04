using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FibonacciTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Fibonacci_Methods();
            Console.ReadLine();
        }

        delegate long Fib(int N, long[] D = null);
        static Fib[] Fibonaci_Methods = { DOWN, UP, Divide_Conquer };
        static int[] D, D1;
        static int sum;

        //divide-and-conquer
        static long Divide_Conquer(int N, long[] D = null)
        {
            if (N == 0)
                return 1;
            else if (N == 1)
                return 1;
            else
                return Divide_Conquer(N - 2) + Divide_Conquer(N - 1);
        }

        //bottom-down method
        static long DOWN(int N, long[] D = null)
        {
            if (D[N] == 0)
                if (N == 0) D[N] = 0;
                else if (N == 1) D[N] = 1;
                else D[N] = DOWN(N - 1, D) + DOWN(N - 2, D);
            return D[N];
        }

        //bottom-up method
        static long UP(int N, long[] D)
        {
            for (int i = 2; i < N; i++)
                D[i] = D[i - 1] + D[i - 2];
            return D[N];
        }
        static double Bine(int N, long[] D = null)
        {
            double phi1 = Math.Pow(((1 + Math.Sqrt(5)) * 0.5), N - 1);
            double phi2 = Math.Pow(((1 - Math.Sqrt(5)) * 0.5), N - 1);
            return (phi1 - phi2) / Math.Sqrt(5);
        }
        static void Fibonacci_Methods()
        {
            Console.Write("Set Size : ");
            int size = int.Parse(Console.ReadLine()) + 1;
            long[] D = new long[size];
            long f = 0;
            D[0] = 0;
            D[1] = 1;
            Stopwatch Time = new Stopwatch();
            //------------------- Бине --------------------
            Time.Reset(); Time.Start();
            double g = Bine(size);
            Console.Write($"{"Bine",14} F ={g,7:F0}");
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write($"| Elapsed ={ Time.ElapsedTicks,7} Ticks\n");
            Console.ResetColor();
            Time.Stop();
            //---------------------------------------------
            foreach (var M in Fibonaci_Methods)
            {
                Time.Reset(); Time.Start();
                if (M.Method.Name == "DOWN" || M.Method.Name == "Divide_Conquer") f = M(--size, D);
                f = M(size, D);
                Time.Stop();
                Console.Write($"{M.Method.Name,14} F ={f,7}");
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.Write($"| Elapsed ={ Time.ElapsedTicks,7} Ticks\n");
                Console.ResetColor();
            }
        }
        //---------------------------------------------
    }
}
